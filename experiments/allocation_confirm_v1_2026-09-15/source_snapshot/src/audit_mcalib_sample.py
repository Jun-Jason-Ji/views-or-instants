"""Validate real observation/Qualisys records without executing arbitrary pickle globals."""
import json,pickle,hashlib,datetime
from pathlib import Path
import numpy as np


class ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('datetime','datetime'):datetime.datetime,('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
                 ('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
                 ('numpy._core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
                 ('numpy.core.multiarray','scalar'):np._core.multiarray.scalar,
                 ('numpy._core.multiarray','scalar'):np._core.multiarray.scalar}
        if (module,name) not in allowed:raise ValueError(f'Unsupported pickle global: {module}.{name}')
        return allowed[module,name]


def read(path):
    with path.open('rb') as f:return ArrayOnlyUnpickler(f).load()


def main():
    root=Path(__file__).resolve().parents[1]/'data_external/mcalib_2026-09-13'
    extracted=root/'verified';records=[]
    for record in ('record1','record2','record3'):
        folder=extracted/'mocapAssistedRecord'/record
        metadata=read(folder/'metadata.pkl')
        lines=(folder/'mocapData.tsv').read_text().splitlines()
        header=next(i for i,x in enumerate(lines) if x.startswith('Frame\tTime'))
        values=np.array([[float(v) if v.strip() else np.nan for v in line.split('\t')[:5]] for line in lines[header+1:] if line.strip()])
        finite=np.all(np.isfinite(values[:,2:5]),axis=1)
        obs={}
        for p in folder.glob('centroidsUV*.pkl'):
            data=read(p)
            valid=[x is not None and np.asarray(x).shape==(2,) and np.all(np.isfinite(x)) for x in data]
            obs[p.stem]={'frames':len(data),'finite_visible':int(sum(valid))}
        records.append({'record':record,'trajectory_rows':len(values),'trajectory_valid':int(finite.sum()),
                        'time_range_s':values[[0,-1],1].tolist(),'median_dt_s':float(np.median(np.diff(values[:,1]))),
                        'strictly_monotone_timestamps':bool(np.all(np.diff(values[:,1])>0)),
                        'coordinate_unit':'mm per official tsvReader.py conversion /1000',
                        'metadata':metadata,'observations':obs})
    intrinsic={}
    for p in (extracted/'intrinsicInit').glob('*.pkl'):
        data=read(p);intrinsic[p.stem]={k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in data.items()}
    def serial(x):
        if isinstance(x,datetime.datetime):return x.isoformat()
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    result={'records':records,'camera_intrinsics':intrinsic,'complete_raw_video_archive_downloaded':False,
            'raw_rgb_prefix_available':(root/'raw_prefix_manifest.json').exists(),
            'extrinsic_correspondence_check_available':(root/'correspondence_validation.json').exists(),'benchmark_record_extracted':False,
            'archive_sha256':hashlib.sha256((root/'MCalib_compactExtractionUV.7z').read_bytes()).hexdigest()}
    (root/'sample_audit.json').write_text(json.dumps(result,indent=2,default=serial))
    print('Verified',len(records),'records and',len(intrinsic),'intrinsic files')


if __name__=='__main__':main()
