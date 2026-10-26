import argparse, json, random
from pathlib import Path
import yaml
p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/terrain.yaml')
p.add_argument('--out',default='outputs/terrain_manifest.json')
a=p.parse_args()
c=yaml.safe_load(Path(a.config).read_text())
rng=random.Random(c['seed'])
t=c['terrain']
rows=[]
for i,kind in enumerate(t['types']):
    item={'terrain_id':i,'kind':kind,'seed':c['seed']}
    if kind=='gap': item['gap_width_m']=rng.uniform(*t['gap_width_m'])
    if kind=='rough': item['height_amplitude_m']=rng.uniform(*t['rough_height_m'])
    if kind=='unseen_stairs':
        item['riser_m']=rng.uniform(*t['stair_riser_m'])
        item['tread_m']=rng.uniform(*t['stair_tread_m'])
    if kind=='mixed': item['weights']=t['mixed_weights']
    rows.append(item)
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'status':'proposed_terrain_parameters_not_simulated','terrains':rows},indent=2))
print(f'Wrote {out} ({len(rows)} terrain parameter sets; no physics simulation executed)')
