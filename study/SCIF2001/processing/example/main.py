import matplotlib.pyplot as plt
import readPTU
import os

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


def dump_header(ptu_file: readPTU.PTUfile, filename: str):
    line = "{:30s} %s {:8}  {:12} "
    with open(filename, "w") as file:
        for key, tag in ptu_file.tags.items():
            value_fmt = "{:>20}"
            if tag["type"] == "tyFloat8":
                value_fmt = "{:20.4g}"
            endline = "\n"
            if tag["type"] == "tyAnsiString":
                endline = tag["data"] + "\n"
            file.write(
                (line % value_fmt).format(
                    key, tag["value"], tag["idx"], tag["type"], end=endline
                )
                + "\n"
            )


files = os.listdir("data")
for file in files:
    run, extension = os.path.splitext(file)
    if extension != ".ptu":
        continue
    try:
        ptu_file = readPTU.PTUfile(f"data/{file}")
    except:
        continue

    dump_header(ptu_file, f"{OUT_DIR}/{run}.txt")

    ptu_meas = readPTU.PTUmeasurement(ptu_file)

    # timetrace
    print(f"{run}", end=" ")
    print(ptu_meas.meas.num_records)
    x1, y1, recnum1 = ptu_meas.timetrace(channel=1)
    print("break")
    x2, y2, recnum2 = ptu_meas.timetrace(channel=2)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x1, y1, color="purple")
    ax.plot(x2, y2, color="seagreen")
    ax.set_title(f"Timetrace for {run}")
    ax.set_xlabel("Time Bucket Index")
    ax.set_ylabel("Photon Count")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{run}-tt.png")

    # g(2) correlation
    hist_x, hist_y = ptu_meas.calculate_g2(
        correlation_window=50000 * 1e-12,
        resolution=600 * 1e-12,
        mode="symmetric",
        channel_start=1,
        channel_stop=2,
    )
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(hist_x, hist_y, color="purple")
    ax.set_title(r"$g^{(2)}$ Correlation for " + f"{run}")
    ax.set_xlabel("Time Bucket Index")
    ax.set_ylabel("Photon Couples Count")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{run}-g2.png")

    print()
