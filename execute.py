import __utilities as ut
import pandas as pd
import warnings

warnings.simplefilter("ignore")

filenames = [
    'DRComparison-Baron.h5',
    'DRComparison-Darmanis.h5',
    'DRComparison-GSE115189Silver.h5',
    'DRComparison-Zhengmix4eq.h5',
    'mat-GSE41265.h5',
    'mat-GSE59739.h5',
    'mat-GSE67602a.h5',
    'mat-GSE70844.h5',
    'mat-GSE74596.h5',
    'mat-GSE75110.h5',
    'scRNAseq-AztekinTailData.h5',
    'scRNAseq-BachMammaryData.h5',
    'scRNAseq-CampbellBrainData.h5',
    'scRNAseq-MarquesBrainData.h5',
    'scRNAseq-ShekharRetinaData.h5',
    'scRNAseq-ZeiselBrainData.h5',
    # 'scRNAseq-ZhaoImmuneLiverData.h5', # Big dataset over 100MB GitHub limit
    'scRNAseq-ZilionisLungData-Mouse.h5',
]


final_results = pd.DataFrame()
for data in filenames:
    print(data)
    results = ut.RNA_seq(data)

    for i in results:
        results[i].update({"aug": i, 'name': filenames})

    results = pd.concat([pd.Series(results[i]) for i in results], axis=1)
    results.columns = [11]  # [7, 11, 21]

    final_results = pd.concat([final_results, results], axis=0)

final_results.to_csv("final_results.csv")
