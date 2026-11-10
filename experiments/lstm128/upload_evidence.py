"""Upload local SHOW-69 artifacts with configured Plane credentials; never print secrets."""
from pathlib import Path
import hashlib
import json
import mimetypes
import tomllib
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parents[2]
CFG=tomllib.loads(Path('/home/jeonchanwook/.codex/config.toml').read_text())['mcp_servers']['plane']['env']
BASE=CFG['PLANE_BASE_URL'].rstrip('/')
PROJECT='ad33ce38-5b24-44a1-9f71-68352c17187f'
ISSUE='819df671-a00e-41d6-b346-d404bd6b51e9'
API=f"{BASE}/api/v1/workspaces/{CFG['PLANE_WORKSPACE_SLUG']}/projects/{PROJECT}/work-items/{ISSUE}/attachments/"
manifest=ROOT/'experiments/lstm128/uploads.json'
records=json.loads(manifest.read_text()) if manifest.exists() else []


def api(method,url,body=None):
    request=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,
        headers={'X-Api-Key':CFG['PLANE_API_KEY'],'Content-Type':'application/json'},method=method)
    with urllib.request.urlopen(request,timeout=30) as response:
        raw=response.read();return json.loads(raw) if raw else None


for file in ['comparison.png','learning-curves.png','policy-rollouts.png','comparison.json']:
    name='show69-'+file
    if any(r['name']==name and r['status']=='uploaded' for r in records):continue
    path=ROOT/'experiments/lstm128/results'/file;payload=path.read_bytes()
    mime=mimetypes.guess_type(name)[0] or 'application/octet-stream'
    raw=api('POST',API,{'name':name,'type':mime,'size':len(payload)})
    asset=raw['asset_id'];boundary='CodexPlane'+uuid.uuid4().hex
    parts=[]
    for k,v in raw['upload_data'].get('fields',{}).items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: {mime}\r\n\r\n'.encode(),payload,f'\r\n--{boundary}--\r\n'.encode()])
    request=urllib.request.Request(raw['upload_data']['url'],data=b''.join(parts),headers={
        'Content-Type':'multipart/form-data; boundary='+boundary},method='POST')
    with urllib.request.urlopen(request,timeout=60) as response:response.read()
    api('PATCH',API+asset+'/',{'is_uploaded':True})
    record={'name':name,'asset_id':asset,'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'status':'uploaded'}
    records.append(record);manifest.write_text(json.dumps(records,indent=2))
    print(json.dumps(record),flush=True)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None


import urllib.error
opener=urllib.request.build_opener(NoRedirect)
for record in records:
    request=urllib.request.Request(API+record['asset_id']+'/',headers={'X-Api-Key':CFG['PLANE_API_KEY']})
    try:
        with opener.open(request,timeout=30) as response:downloaded=response.read()
    except urllib.error.HTTPError as error:
        if error.code not in [301,302,303,307,308]:raise
        # A storage request must not inherit the Plane API authentication header.
        with urllib.request.urlopen(error.headers['Location'],timeout=30) as response:downloaded=response.read()
    assert hashlib.sha256(downloaded).hexdigest()==record['sha256'],record['name']
    record['download_sha256_verified']=True
manifest.write_text(json.dumps(records,indent=2))
print(json.dumps({'verified_attachment_count':len(records)}))
