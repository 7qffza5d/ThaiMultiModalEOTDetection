import glob, os
tg_dir = r"D:\Projects\Ogma\ThaiMultiModalEOTDetection\rawData\text\textgrid"
tg  = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(tg_dir, "*.TextGrid"))}
out = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob("timestamped/*.parquet")}
print(len(tg), len(out))
print(sorted(tg - out))