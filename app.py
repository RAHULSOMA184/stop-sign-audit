"""Batch Street View STOP-sign approach audit. Evidence-based, review-first."""
import io, math, os, json, time, hashlib, re, gzip
from pathlib import Path
from datetime import datetime
import osmnx as ox
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
ox.settings.use_cache = True
ox.settings.cache_folder = str(ROOT / 'cache')
ox.settings.requests_timeout = 35
st.set_page_config(page_title='STOP Sign Detection',layout='wide',initial_sidebar_state='collapsed')
@st.cache_data(ttl=86400, show_spinner=False)
def geocode(address, key):
    r=requests.get('https://maps.googleapis.com/maps/api/geocode/json',params={'address':address,'key':key},timeout=25);r.raise_for_status()
    d=r.json()
    if d.get('status')!='OK':raise RuntimeError(f"Geocode: {d.get('status')} {d.get('error_message','')}")
    return d['results'][0]['geometry']['location']

def distance(a,b):
    lat1,lon1=a;lat2,lon2=b;R=6371000
    p1,p2=map(math.radians,[lat1,lat2]);dp=p2-p1;dl=math.radians(lon2-lon1)
    h=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(min(1,math.sqrt(h)))

def bearing(a,b):
    lat1,lon1=map(math.radians,a);lat2,lon2=map(math.radians,b);dl=lon2-lon1
    return (math.degrees(math.atan2(math.sin(dl)*math.cos(lat2),math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(dl)))+360)%360

def road_name(x):
    if isinstance(x,list):return ', '.join(str(z) for z in x)
    return str(x or '')



def graph_from_saved_overpass(cache_dir, lat, lon):
    """Rebuild a road graph from an existing OSMnx Overpass JSON response.

    Works offline; does not rely on OSMnx's URL-sensitive cache lookup.
    """
    import json
    import networkx as nx
    from pathlib import Path

    files = sorted(Path(cache_dir).glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f'No OSMnx JSON cache found in {cache_dir}')
    errors = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding='utf-8'))
            elements = data.get('elements', [])
            nodes = {e['id']: e for e in elements if e.get('type') == 'node'
                     and 'lat' in e and 'lon' in e}
            if not nodes:
                continue
            # The saved response must cover this intersection, not a different town.
            closest = min(distance((lat, lon), (n['lat'], n['lon'])) for n in nodes.values())
            if closest > 90:
                errors.append(f'{file.name}: nearest node {closest:.0f}m away')
                continue
            G = nx.MultiDiGraph()
            for node_id, n in nodes.items():
                G.add_node(node_id, y=n['lat'], x=n['lon'])
            for way in elements:
                if way.get('type') != 'way':
                    continue
                tags = way.get('tags', {})
                if 'highway' not in tags:
                    continue
                sequence = way.get('nodes', [])
                for a, b in zip(sequence, sequence[1:]):
                    if a in nodes and b in nodes:
                        G.add_edge(a, b, **tags)
                        G.add_edge(b, a, **tags)
            if G.number_of_edges() == 0:
                continue
            return G, file.name
        except Exception as exc:
            errors.append(f'{file.name}: {exc}')
    raise RuntimeError('No suitable cached OSM road network: ' + '; '.join(errors[:5]))


# A single streaming pass over the statewide PBF extracts only ways whose road
# names occur in the uploaded spreadsheet. No Overpass requests are made.
PBF = ROOT / 'data' / 'ma_six_city_roads.json.gz'

def normalize_name(value):
    # Match punctuation/spelling variants: Oneil, O'Neil, O Neil.
    return re.sub(r'[^a-z0-9]', '', str(value or '').casefold())

def name_matches(osm_value, requested):
    target = normalize_name(requested)
    if isinstance(osm_value, list):
        return any(name_matches(v, requested) for v in osm_value)
    return normalize_name(osm_value) == target

@st.cache_resource(show_spinner='Loading preprocessed road geometry...')
def load_local_ways(pbf_path, pbf_mtime, requested_names):
    with gzip.open(pbf_path, 'rt', encoding='utf-8') as f:
        payload = json.load(f)
    if payload.get('format') != 'stop-sign-roads-v1':
        raise ValueError('Unsupported road dataset format. Re-run preprocess_ma_roads.py.')
    wanted = set(requested_names)
    return [(name, points) for name, points in payload['ways'] if normalize_name(name) in wanted]

def local_graph(lat, lon, road_a, road_b, ways):
    import networkx as nx
    wanted = {normalize_name(road_a), normalize_name(road_b)}
    G = nx.MultiDiGraph()
    for name, points in ways:
        if normalize_name(name) not in wanted:
            continue
        for a, b in zip(points, points[1:]):
            pa, pb = (a[1], a[2]), (b[1], b[2])
            if min(distance((lat,lon),pa),distance((lat,lon),pb)) > 280:
                continue
            for nid, y, x in (a,b):
                G.add_node(nid,y=y,x=x)
            G.add_edge(a[0],b[0],name=name)
            G.add_edge(b[0],a[0],name=name)
    if not G.number_of_edges():
        raise RuntimeError('No matching named roads within 280m of geocoded location. Check Google geocode and OSM road names.')
    return G

def load_road_graph(lat, lon, road_a, road_b):
    if not PBF.is_file():
        raise FileNotFoundError(f'Preprocessed road dataset not found: {PBF}. Run preprocess_ma_roads.py first.')
    names = st.session_state.get('batch_road_names')
    if not names:
        names = tuple(sorted({normalize_name(road_a),normalize_name(road_b)}))
    ways = load_local_ways(str(PBF), PBF.stat().st_mtime_ns, names)
    return local_graph(lat,lon,road_a,road_b,ways), 'LOCAL preprocessed road geometry'

@st.cache_data(ttl=86400,show_spinner='Retrieving OSM road geometry...')
def get_approaches(lat,lon,road_a,road_b):
    import osmnx as ox
    G, source = load_road_graph(lat, lon, road_a, road_b)
    # An unsimplified graph includes every shape point. Locate the closest point where
    # both named streets meet, then follow outgoing street branches through shape nodes.
    candidates=[]
    for n,attrs in G.nodes(data=True):
        p=(attrs['y'],attrs['x']);d=distance((lat,lon),p)
        if d>90:continue
        neighbors=set(G.successors(n))|set(G.predecessors(n))
        names=set()
        for nb in neighbors:
            for u,v in [(n,nb),(nb,n)]:
                if G.has_edge(u,v):
                    for e in G[u][v].values():
                        name=e.get('name','')
                        if name_matches(name,road_a):names.add('a')
                        if name_matches(name,road_b):names.add('b')
        if len(names)==2:candidates.append((d,n))
    if not candidates:raise RuntimeError('Could not find an OSM node connecting both requested roads within 90m. Use the manual camera editor or check intersection coordinates.')
    center_node=min(candidates)[1]
    center=(G.nodes[center_node]['y'],G.nodes[center_node]['x'])
    # Walk away from the intersection on each branch. Unsimplified graph makes
    # following a curved road more accurate than a straight compass offset.
    first=set(G.successors(center_node))|set(G.predecessors(center_node))
    branches=[]
    for nxt in first:
        names=[]
        for u,v in [(center_node,nxt),(nxt,center_node)]:
            if G.has_edge(u,v):names += [road_name(e.get('name','')) for e in G[u][v].values()]
        road=road_a if any(name_matches(s,road_a) for s in names) else road_b if any(name_matches(s,road_b) for s in names) else None
        if not road:continue
        coords=[center,(G.nodes[nxt]['y'],G.nodes[nxt]['x'])]
        prev,cur=center_node,nxt
        for _ in range(120):
            if sum(distance(coords[j],coords[j+1]) for j in range(len(coords)-1))>=65:break
            neighbors=(set(G.successors(cur))|set(G.predecessors(cur)))-{prev}
            options=[]
            for nb in neighbors:
                match=False
                for u,v in [(cur,nb),(nb,cur)]:
                    if G.has_edge(u,v):
                        for e in G[u][v].values():
                            if name_matches(e.get('name',''),road):match=True
                if match:options.append(nb)
            if len(options)!=1:break
            prev,cur=cur,options[0];coords.append((G.nodes[cur]['y'],G.nodes[cur]['x']))
        if len(coords)<2:continue
        branches.append({'road':road,'coords':coords,'outbound':bearing(center,coords[min(1,len(coords)-1)])})
    # Deduplicate nearly identical branches caused by parallel/one-way OSM edges.
    distinct=[]
    for b in branches:
        if not any(b['road']==z['road'] and abs((b['outbound']-z['outbound']+180)%360-180)<25 for z in distinct):distinct.append(b)
    distinct=sorted(distinct,key=lambda x:(x['road'],x['outbound']))
    return center,distinct,source

def sample_branch(coords,meters):
    remaining=meters
    for a,b in zip(coords,coords[1:]):
        length=distance(a,b)
        if length>=remaining and length>0:
            t=remaining/length
            return (a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)
        remaining-=length
    return coords[-1]

def metadata(lat,lon,key):
    r=requests.get('https://maps.googleapis.com/maps/api/streetview/metadata',params={'location':f'{lat:.7f},{lon:.7f}','radius':18,'source':'outdoor','key':key},timeout=25);r.raise_for_status();return r.json()

def fetch_image(pano,heading,key):
    r=requests.get('https://maps.googleapis.com/maps/api/streetview',params={'pano':pano,'size':'640x640','heading':round(heading,1),'pitch':0,'fov':80,'key':key},timeout=30);r.raise_for_status()
    if not r.headers.get('Content-Type','').startswith('image/'):raise RuntimeError('Google did not return an image')
    return Image.open(io.BytesIO(r.content)).convert('RGB')

@st.cache_resource(show_spinner='Loading local YOLO model (first run downloads weights)...')
def model():
    from ultralytics import YOLO
    return YOLO('yolo11n.pt')

def detect(img,conf):
    result=model().predict(img,conf=conf,verbose=False)[0]
    matches=[]
    for b in result.boxes:
        label=result.names[int(b.cls[0])]
        if label.lower()=='stop sign':
            matches.append({'confidence':round(float(b.conf[0]),3),'box':[round(float(z)) for z in b.xyxy[0].tolist()]})
    marked=img.copy();draw=ImageDraw.Draw(marked)
    for m in matches:draw.rectangle(m['box'],outline='red',width=4)
    return matches,marked



def process_approach(branch, center, key, threshold, max_snap, age_months, offsets, side_views=False):
    """Returns evidence metadata and in-memory images. No Street View bytes written to disk."""
    views=[];seen=set();success=False
    for d in offsets:
        target=sample_branch(branch['coords'],d)
        try:
            m=metadata(*target,key)
            if m.get('status')!='OK':
                views.append({'distance':d,'error':m.get('status','No panorama')});continue
            pano=m['pano_id'];actual=(m['location']['lat'],m['location']['lng'])
            snap=round(distance(target,actual),1)
            if pano in seen:
                views.append({'distance':d,'error':'Duplicate panorama — skipped'});continue
            seen.add(pano)
            heading=bearing(actual,center)
            angles=[0,-22,22] if side_views else [0]
            for delta in angles:
                h=(heading+delta)%360
                img=fetch_image(pano,h,key)
                matches,marked=detect(img,threshold)
                v={'distance':d,'date':m.get('date','Unknown'),'pano':pano,
                   'snap_m':snap,'actual':actual,'heading':round(h,1),
                   'delta':delta,'matches':matches,'image':marked,
                   'valid_snap':snap<=max_snap}
                views.append(v)
                # Stop only for a well-positioned, forward-looking panorama. The sign
                # still requires human approach-control verification.
                if matches and v['valid_snap'] and delta==0:
                    success=True
                    break
            if success:break
        except Exception as e:
            views.append({'distance':d,'error':str(e)})
    return views

def age_flag(date_str, months):
    if not date_str or date_str=='Unknown':return 'Unknown date'
    try:
        d=datetime.strptime(date_str[:7],'%Y-%m')
        now=datetime.now()
        age=(now.year-d.year)*12+now.month-d.month
        return 'Older than threshold' if age>months else 'Within threshold'
    except ValueError:return 'Unknown date'

def classify(views,max_snap,age_months):
    valid=[v for v in views if 'image' in v]
    credible=[v for v in valid if v['snap_m']<=max_snap]
    hits=[v for v in credible if v['matches']]
    best=max(hits,key=lambda v:max(m['confidence'] for m in v['matches'])) if hits else None
    if hits:status='STOP candidate — review sign orientation'
    elif not credible:status='REVIEW — no suitable panorama'
    else:status='NO STOP detected — review required'
    flags=[]
    if not credible:flags.append('No panoramas within snap tolerance')
    if any(v['snap_m']>max_snap for v in valid):flags.append('Some panoramas snapped too far')
    if any(age_flag(v['date'],age_months)!='Within threshold' for v in credible):flags.append('Old or unknown imagery date')
    if any('error' in v for v in views):flags.append('Missing/duplicate imagery at some offsets')
    if not hits:flags.append('Absence not established')
    return status,best,flags


COLUMNS = ['Major Rd Name','Minor Rd Name','Leg Count','Town']
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
GEOM = ROOT / 'geometry'
GEOM.mkdir(exist_ok=True)

def slug(row):
    label = '|'.join(str(row[k]).strip().lower() for k in COLUMNS)
    return hashlib.sha256(label.encode()).hexdigest()[:16]

def geometry_for(row, lat, lon):
    file = GEOM / (slug(row)+'.json')
    if file.exists():
        data=json.loads(file.read_text(encoding='utf-8'))
        return tuple(data['center']),data['branches'],'Saved intersection geometry'
    center,branches,source=get_approaches(lat,lon,str(row['Major Rd Name']).strip(),str(row['Minor Rd Name']).strip())
    file.write_text(json.dumps({'center':center,'branches':branches},indent=2),encoding='utf-8')
    return center,branches,source

def audit_one(row,key,threshold,max_snap,age_months,offsets,side_views):
    road_a=str(row['Major Rd Name']).strip()
    road_b=str(row['Minor Rd Name']).strip()
    town=str(row['Town']).strip()
    legs=int(row['Leg Count'])
    address=f'{road_a} & {road_b}, {town}, Massachusetts, USA'
    common={'intersection_id':slug(row),'intersection':address,'major_road':road_a,
            'minor_road':road_b,'town':town,'expected_legs':legs}
    loc=geocode(address,key)
    center,branches,source=geometry_for(row,loc['lat'],loc['lng'])
    records=[]
    evidence={}
    for i,b in enumerate(branches,1):
        views=process_approach(b,center,key,threshold,max_snap,age_months,offsets,side_views)
        status,best,flags=classify(views,max_snap,age_months)
        evidence[i]=views  # images remain in memory, never saved to disk
        credible=[v for v in views if 'image' in v and v['snap_m']<=max_snap]
        images=[v for v in views if 'image' in v]
        records.append({**common,'approach':i,'road':b['road'],
            'incoming_heading_deg':round(bearing(b['coords'][-1],center),1),
            'ai_assessment':status,'stop_candidate':bool(best),
            'best_distance_m':best['distance'] if best else None,
            'best_confidence':max(x['confidence'] for x in best['matches']) if best else None,
            'best_date':best['date'] if best else '',
            'best_pano_id':best['pano'] if best else '',
            'best_heading':best['heading'] if best else None,
            'best_lat':best['actual'][0] if best else None,
            'best_lon':best['actual'][1] if best else None,
            'views_checked':len(images),'credible_views':len(credible),
            'image_dates':'; '.join(sorted(set(v['date'] for v in images))),
            'flags':'; '.join(flags),'geometry_source':source,
            'review_result':'Not reviewed'})
    candidates=sum(r['stop_candidate'] for r in records)
    unresolved=sum(r['credible_views']==0 for r in records)
    mismatch=len(branches)!=legs
    summary={**common,'found_legs':len(branches),'stop_candidates':candidates,
        'no_stop_detected':len(branches)-candidates-unresolved,
        'unresolved_approaches':unresolved,
        'leg_count_match':not mismatch,'status':('REVIEW — leg mismatch' if mismatch else
            'REVIEW — missing imagery' if unresolved else 'AI COMPLETE — human review needed'),
        'error':'','geometry_source':source}
    return summary,records,evidence

def read_input(upload):
    if upload.name.lower().endswith('.csv'):
        df=pd.read_csv(upload)
    else:
        df=pd.read_excel(upload)
    df.columns=[str(c).strip() for c in df.columns]
    missing=[c for c in COLUMNS if c not in df.columns]
    if missing:raise ValueError('Missing columns: '+', '.join(missing))
    df=df[COLUMNS].copy().dropna(how='all')
    df=df.dropna(subset=COLUMNS)
    df['Leg Count']=pd.to_numeric(df['Leg Count'],errors='raise').astype(int)
    if not df['Leg Count'].between(2,8).all():raise ValueError('Leg Count must be between 2 and 8')
    return df

def export_book(summaries,approaches):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as writer:
        pd.DataFrame(summaries).to_excel(writer,sheet_name='Intersections',index=False)
        pd.DataFrame(approaches).to_excel(writer,sheet_name='Approaches',index=False)
        pd.DataFrame([{'note':'STOP candidate means YOLO found a STOP sign in a suitable image; sign orientation still needs review.'},
            {'note':'No STOP detected does not establish absence.'},
            {'note':'Imagery is displayed during processing but is not stored in this workbook.'},
            {'note':'Google Street View content usage is subject to Google Maps Platform terms.'}]).to_excel(writer,sheet_name='Read me',index=False)
    return buf.getvalue()

st.title('STOP Sign Batch Audit — V3.2 (Visual Review)')
st.caption('Upload intersections → road-aligned approaches → adaptive Street View → YOLO → Excel')
st.warning('AI results are screening findings, not proof of current traffic-control inventory. Inspect sign orientation and flagged imagery before relying on results.')
key=st.text_input('Google Maps API key',value=os.getenv('GOOGLE_MAPS_API_KEY',''),type='password')
upload=st.file_uploader('Intersections.xlsx (or CSV)',type=['xlsx','csv'])
with st.expander('Settings',expanded=True):
    a,b,c=st.columns(3)
    threshold=a.slider('YOLO confidence',0.10,0.80,0.25,0.05)
    max_snap=b.slider('Max panorama snap distance (m)',5,40,18)
    age_months=c.slider('Flag imagery older than (months)',6,72,24,6)
    offsets=st.multiselect('Adaptive camera distances (m)',[10,15,20,25,30,35,40,45,50,60],default=[20,30,40])
    side_views=st.checkbox('Check side views ±22° (more API calls)',False)
    st.caption('Road geometry is loaded from data/ma_six_city_roads.json.gz. No Overpass required.')
    limit=st.number_input('Intersections to process this run (start with 5)',min_value=1,max_value=10000,value=5)

if PBF.is_file():
    st.success(f'Preprocessed road file found: {PBF.name} ({PBF.stat().st_size / 1024**2:.0f} MB)')
else:
    st.error(f'Preprocessed road file missing: {PBF}')

if upload:
    try:
        data=read_input(upload)
        st.write(f'{len(data)} valid intersection rows found')
        st.dataframe(data.head(15),hide_index=True,width=stretch)
    except Exception as exc:
        st.error(str(exc));data=None
else:data=None

if st.button('Run batch',type='primary',disabled=(data is None or not key or not offsets or not PBF.is_file())):
    summaries=[];approaches=[];evidence_by_intersection={}
    progress=st.progress(0)
    status=st.empty()
    subset=data.head(int(limit))
    st.session_state['batch_road_names']=tuple(sorted({normalize_name(name) for col in ('Major Rd Name','Minor Rd Name') for name in subset[col]}))
    for idx,(_,series) in enumerate(subset.iterrows()):
        row=series.to_dict()
        status.write(f"Processing {idx+1}/{len(subset)}: {row['Major Rd Name']} @ {row['Minor Rd Name']}, {row['Town']}")
        try:
            s,details,evidence=audit_one(row,key,threshold,max_snap,age_months,offsets,side_views)
            summaries.append(s);approaches.extend(details)
            evidence_by_intersection[slug(row)]=evidence
        except Exception as exc:
            summaries.append({'intersection_id':slug(row),
                'intersection':f"{row['Major Rd Name']} & {row['Minor Rd Name']}, {row['Town']}",
                'major_road':row['Major Rd Name'],'minor_road':row['Minor Rd Name'],
                'town':row['Town'],'expected_legs':int(row['Leg Count']),
                'found_legs':None,'stop_candidates':None,'no_stop_detected':None,
                'unresolved_approaches':None,'leg_count_match':False,
                'status':'FAILED — review','error':str(exc)[:900]})
        # Save metadata-only progress after each intersection; resumable evidence workbook
        (OUT/'latest_summary.json').write_text(json.dumps(summaries,indent=2,default=str),encoding='utf-8')
        (OUT/'latest_approaches.json').write_text(json.dumps(approaches,indent=2,default=str),encoding='utf-8')
        progress.progress((idx+1)/len(subset))
    st.session_state['last_summary']=summaries
    st.session_state['last_approaches']=approaches
    st.session_state['last_evidence']=evidence_by_intersection
    status.success('Batch finished. Review failures and AI candidates below.')

if 'last_summary' in st.session_state:
    summaries=st.session_state['last_summary'];approaches=st.session_state['last_approaches']
    st.subheader('Intersection results')
    st.dataframe(pd.DataFrame(summaries),width=stretch,hide_index=True)
    st.subheader('Approach results')
    st.dataframe(pd.DataFrame(approaches),width=stretch,hide_index=True)
    st.subheader('Visual evidence & human verification')
    st.caption('Expand an intersection and approach to inspect Street View frames with YOLO boxes. Photos are retained only in this Streamlit session, not exported to Excel or saved to disk. Rerunning the batch replaces them.')
    evidence_by=st.session_state.get('last_evidence',{})
    if not evidence_by:
        st.warning('No images are available in this session. Run the batch using this version to capture visual evidence. The previous Excel file does not contain the images.')
    @st.fragment
    def review_fields(record, sid, approach):
        options=['Not reviewed','STOP confirmed','No STOP visible','Unclear / wrong approach']
        previous=record.get('review_result','Not reviewed')
        choice=st.selectbox('Your verification',options,index=options.index(previous) if previous in options else 0,key=f'review_{sid}_{approach}')
        record['review_result']=choice
        record['review_notes']=st.text_input('Review notes (optional)',value=record.get('review_notes',''),key=f'notes_{sid}_{approach}')

    @st.fragment(run_every='2s')
    def live_review_count(sid, related):
        reviewed=sum(r.get('review_result','Not reviewed')!='Not reviewed' for r in related)
        st.caption(f'Human verification: {reviewed}/{len(related)} approaches reviewed')

    @st.fragment(run_every='2s')
    def live_download(summaries, approaches):
        st.download_button('Download consolidated Excel',export_book(summaries,approaches),
            'stop_sign_batch_results.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    for summary in summaries:
        sid=summary['intersection_id']
        related=[r for r in approaches if r['intersection_id']==sid]
        if not related:
            continue
        live_review_count(sid, related)
        with st.expander(summary['intersection'], expanded=False):
            for record in related:
                approach=int(record['approach'])
                views=evidence_by.get(sid,{}).get(approach,[])
                with st.expander(f"Approach {approach}: {record['road']} · {record['incoming_heading_deg']}° · {record['ai_assessment']}", expanded=False):
                    if not views:
                        st.warning('No image evidence for this approach in this session.')
                    for view in views:
                        if 'error' in view:
                            st.warning(f"{view['distance']} m: {view['error']}")
                            continue
                        conf=max((hit['confidence'] for hit in view['matches']),default=None)
                        conf_label=f'{conf:.1%}' if conf is not None else 'none'
                        st.write(f"**{view['distance']} m** | heading {view['heading']}° | YOLO STOP confidence: {conf_label} | snap {view['snap_m']} m | imagery {view['date']}")
                        st.image(view['image'],width=520,caption='Red boxes = YOLO STOP sign candidates. Check whether the sign controls THIS incoming approach.')
                        if view['matches']:
                            st.json(view['matches'])
                    review_fields(record, sid, approach)
    st.caption('The downloaded Approaches sheet includes your verification and notes. AI classifications remain unchanged for comparison.')
    live_download(summaries, approaches)
    st.info('Rows marked FAILED need review of geocoding, OSM road-name matching, or Street View API availability. No Overpass is used.')
