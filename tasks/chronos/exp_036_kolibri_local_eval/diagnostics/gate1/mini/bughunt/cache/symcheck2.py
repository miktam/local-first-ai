exec(open("symcheck.py").read().split("T = 16384 + 37")[0])
rng = np.random.default_rng(9)
sched = []
while sum(sched) < 17000:
    sched += [int(rng.choice([1, 2, 7, 63, 64, 65, 255, 256, 257, 511, 512, 513, 514, 1025, 2048]))] * int(rng.integers(1, 4))
run(lambda T: sched, sum(sched), f"mixed decode/prefill transitions ({len(sched)} chunks)")
run(lambda T: [600, 1, 1, 64, 1, 513, 1, 2, 514, 1] * 30, 1698 * 30, "alternating 1 and >1")
