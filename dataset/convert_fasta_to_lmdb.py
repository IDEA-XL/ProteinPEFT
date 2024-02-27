import os
import os.path as osp
from tqdm import tqdm
import json
import lmdb
import argparse

# read fasta
def read_fasta(fasta_file):
    sequence_labels, sequence_strs = [], []
    cur_seq_label = None
    buf = []
    
    def _flush_current_seq():
        nonlocal cur_seq_label, buf
        if cur_seq_label is None:
            return
        sequence_labels.append(cur_seq_label)
        sequence_strs.append("".join(buf))
        cur_seq_label = None
        buf = []

    with open(fasta_file, "r") as infile:
        for line_idx, line in enumerate(infile):
            if line.startswith(">"):  # label line
                _flush_current_seq()
                line = line[1:].strip()
                if len(line) > 0:
                    cur_seq_label = line
                else:
                    cur_seq_label = f"seqnum{line_idx:09d}"
            else:  # sequence line
                buf.append(line.strip())
    
    _flush_current_seq()
    return sequence_labels, sequence_strs
    

if __name__ == "__main__":
    """
    python -f /path/to/fasta -db /path/to/lmdb [--lock]
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta_path", type=str, required=True)
    parser.add_argument("-db", "--lmdb_path", type=str, required=True)
    parser.add_argument("--lock", default=False, action="store_true")
    args = parser.parse_args()
    
    assert osp.exists(args.fasta_path), f"fasta file {args.fasta_path} not found"
    if osp.isdir(args.lmdb_path):
        args.lmdb_path = osp.join(args.lmdb_path, "data.lmdb")
    if not args.lmdb_path.endswith(".mdb") or not args.lmdb_path.endswith(".lmdb"):
        lmdb_dir = osp.dirname(args.lmdb_path)
        assert osp.isdir(lmdb_dir), f"lmdb directory {lmdb_dir} not found"
    
    seq_labels, seq_strs = read_fasta(args.fasta_path)

    _10TB = 10995116277760
    env_new = lmdb.open(
        args.lmdb_path,
        subdir=False,
        readonly=False,
        lock=args.lock,
        readahead=False,
        meminit=False,
        map_size=_10TB,
        max_readers=1,
    )
    txn_writer = env_new.begin(write=True)
    i = 0
    error_cnt = 0
    for i, (label, seq) in tqdm(enumerate(zip(seq_labels, seq_strs))):
        uniref50_ID = label.split(" ")[0].split("_")[1]
        try:
            key = f"{i:09d}".encode("utf-8")
            value = dict(
                description=uniref50_ID,
                seq=seq,
            )
            value = json.dumps(value).encode("utf-8")
            txn_writer.put(key, value)
        except Exception as e:
            print(e)
            error_cnt += 1
            continue
        if i % 10000 == 0:
            txn_writer.commit()
            txn_writer = env_new.begin(write=True)
    
    # write in "length" key for the length of the dataset
    txn_writer.put("length".encode("utf-8"), str(i).encode("utf-8"))
    txn_writer.commit()
    
    # Finish writing
    env_new.sync()
    env_new.close()
    
    print(f"Error count: {error_cnt}")
    print(f"Total sequences: {i}")