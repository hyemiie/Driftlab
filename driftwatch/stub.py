import json
from types import SimpleNamespace as NS
from driftwatch.db.models import ColumnRole
from driftwatch.services.ingestion import read_csv
from driftwatch.services.schema_infer import infer_schema
from driftwatch.services.profiling import compute_reference_profile

df = read_csv(open("titanic.csv", "rb").read())
cols = [NS(name=c.name, dtype=c.dtype, role=ColumnRole.feature) for c in infer_schema(df)]
stats = compute_reference_profile(df, cols)
print(json.dumps(stats["Age"], indent=2))
print(json.dumps(stats["Sex"], indent=2))