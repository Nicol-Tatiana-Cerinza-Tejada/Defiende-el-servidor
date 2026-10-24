"""Comprobación independiente: evidencia guardada, sin reevaluar test."""
import sys,json,hashlib,urllib.request
from pathlib import Path
import pandas as pd,numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from common import binary_metrics
out=root/'evidencias/revision-natalia';out.mkdir(parents=True,exist_ok=True)
a=json.loads((root/'evidencias/auditoria.json').read_text());m=json.loads((root/'evidencias/metricas-test.json').read_text());s=json.loads((root/'evidencias/seleccion_validacion.json').read_text())
p=pd.read_csv(root/'evidencias/predicciones-test.csv')
assert len(p)==82332 and p.row_id.is_unique
assert (p.y_pred==(p.score_attack>=m['threshold']).astype(int)).all()
assert list(p.row_id)==[f'test:{i}' for i in range(82332)]
calc=binary_metrics(p.y_true,p.y_pred)
for k,v in calc.items(): assert abs(v-m['metrics'][k])<1e-12,(k,v,m['metrics'][k])
assert hashlib.sha256((root/'evidencias/predicciones-test.csv').read_bytes()).hexdigest()==m['predictions_sha256']
e=pd.read_csv(root/'evidencias/experimentos.csv');assert len(e)==4 and e.run_id.nunique()==4 and e.family.nunique()>=2
for _,r in e.iterrows():
 assert (root/'evidencias/mlflow'/r.run_id).is_dir()
 assert r.dvc_raw_md5=='fe93051ae209ffb0128a31e7a478c009.dir'
 assert r.dvc_prepared_md5=='b3533724917eef902710882f831d14e5.dir'
assert e.sort_values('val_macro_f1',ascending=False).iloc[0].exp_id=='E4_hist_gb'
assert s['chosen_threshold']==.55 and s['test_used'] is False
assert a['cleaning_policy']['contradictory_rows_excluded']==940 and a['split']['groups_shared_train_val']==0
hf={}
for v,rev in [('v1','2e405585467e01b576cbfd324488fa07174ef39e'),('v2','4ea2e9e7b2245d27a672e58b81d27574843615f7')]:
 d=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/NicolTcT/defiende-el-servidor/revision/{rev}'))
 names=[x['rfilename'] for x in d['siblings']];assert {'README.md','model.joblib','config.json','requirements.txt'}<=set(names)
 hf[v]={'sha':rev,'files':names}
report={'checked_code_commit':'1e94c07957f8c742b51fb7ce94e6fe50c81fb011','test_rows':len(p),'test_predictions_sha256_matches':True,'test_metrics_recalculated':calc,'experiments':e[['exp_id','run_id','val_macro_f1']].to_dict(orient='records'),'hub_revisions':hf,'audit_declared_contradictory_rows':940,'note':'Independent file-level review; fresh recovery and retraining documented separately. No new test selection or evaluation.'}
(out/'comprobacion-evidencias.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print('EVIDENCIAS OK: 82332 filas, métricas recalculadas, cuatro runs, selección y archivos HF.')
