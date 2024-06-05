Population biases project
====================================================================================================================

Pull the annotated InDiCo data using the following commands 
```bash
cd population-biases
sh pull_data.sh
```

Estimate and extract surprisal and entropy measures
```bash
python src/experiments.py
```

Analysis scripts for Hypotheses 0-2, using combined ET data
```bash
Rscript --vanilla src/analyses_h0-h2.R
```

Analysis scripts for Hypothesis 3, using ET data split into groups
```bash
Rscript --vanilla src/analyses_h3.R
```