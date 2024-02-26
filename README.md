# Data

UniRef50 63,849,054 

* 99% len: 1468

UniRef90 184,520,054

## Preprocess

```bash
# use mmseqs to deduplicate train from valid
mmseqs createdb train.fasta DB/train
mmseqs createdb valid.fasta DB/valid
mmseqs createindex DB/valid tmp

# use s=7 # ETC: 16 hours
mmseqs search DB/train DB/valid DB/train_dup tmp --min-seq-id 0.5 --alignment-mode 3 -s 7 --max-seqs 300  -c 0.8 --cov-mode 0
# use s=5.6 (default) # ETC: 4 hours
mmseqs search DB/train DB/valid DB/train_dup tmp --min-seq-id 0.5 --alignment-mode 3 --max-seqs 300 -c 0.8 --cov-mode 0

# 1. convert to m8 format
 mmseqs convertalis DB/train DB/valid DB/train_dup DB/train_dup.m8
# 2. extract first column (means duplicated train samples)
cut -f1 DB/train_dup.m8 > train_dup.id.txt
sort train_dup.id.txt | uniq > train_dup1.id.txt
rm train_dup.id.txt
mv train_dup1.id.txt train_dup.id.txt
# 3. deduplicate
seqkit grep -v -f train_dup.id.txt train.fasta -o train_dedup.fasta
```

After deduplication.
| s             | # train      | # valid   | # train_dedup |
|---------------|--------------|-----------|---------------|
| default(5.6)  | 63,070,639   | 319,332   | 459,083       |
| 7             | 63,067,918   | 319,332   | 461,804       |

# CodeBase
SaProt https://github.com/westlake-repl/SaProt