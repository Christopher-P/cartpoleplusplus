#!/usr/bin/env python
import argparse, sys, re, json
import numpy as np
from collections import Counter

f_actions= open("/tmp/actions", "w")
f_actions.write("time type x y\n")
f_q_values = open("/tmp/q_values", "w")
f_q_values.write("time net_type q_value\n")
f_episode_len = open("/tmp/episode_stats", "w")
f_episode_len.write("episode len total_reward\n")
f_eval = open("/tmp/eval", "w")
f_eval.write("time steps total_reward\n")
f_batch_num_terminal = open("/tmp/batch_num_terminal", "w")
f_batch_num_terminal.write("time batch_num_terminals\n")
f_gradient_l2_norms = open("/tmp/gradient_l2_norms", "w")
f_gradient_l2_norms.write("time source l2_norm\n")
f_q_loss = open("/tmp/q_loss", "w")
f_q_loss.write("time q_loss\n")

freq = Counter()
emit_freq = {"EVAL": 1, "ACTOR_L2_NORM": 20, "CRITIC_L2_NORM": 20, 
             "Q LOSS": 1, "EXPECTED_Q_VALUES": 1}
def should_emit(tag):
  freq[tag] += 1
  return freq[tag] % (emit_freq[tag] if tag in emit_freq else 100) == 0

n_parse_errors = 0

time = None
for line in sys.stdin:
  line = line.strip()

  if line.startswith("STATS"):
    cols = line.split("\t")
    assert len(cols) == 2, line
    try:
      d = json.loads(cols[1])
      if should_emit("EPISODE_LEN"):
        episode = d["episode"]
        total_reward = d["total_reward"]
        episode_len = d["episode_len"] if "episode_len" in d else total_reward
        time = d["time"]
        f_episode_len.write("%s %s %s\n" % (episode, episode_len, total_reward))
    except ValueError:
      # interleaving output :/
      n_parse_errors += 1

  if time is None:
    continue

  elif "actor gradient l2_norm" in line and should_emit("ACTOR_L2_NORM"):
    norm = re.sub(".*\[", "", line).replace("]", "")
    f_gradient_l2_norms.write("%s actor %s\n" % (time, norm))

  elif "critic gradient l2_norm" in line and should_emit("CRITIC_L2_NORM"):
    norm = re.sub(".*\[", "", line).replace("]", "")
    f_gradient_l2_norms.write("%s critic %s\n" % (time, norm))

  elif line.startswith("ACTIONS") and should_emit("ACTIONS"):
    m = re.match("ACTIONS\t\[(.*), (.*)\]\t\[(.*), (.*)\]", line)
    if m:
      pre_x, pre_y, post_x, post_y = m.groups()
      f_actions.write("%s pre %s %s\n" % (time, pre_x, pre_y))
      f_actions.write("%s post %s %s\n" % (time, post_x, post_y))

  elif line.startswith("EXPECTED_Q_VALUES") and should_emit("EXPECTED_Q_VALUES"):
    cols = line.split(" ")
    assert len(cols) == 3
    assert cols[0] == "EXPECTED_Q_VALUES"
    f_q_values.write("%s main %f\n" % (time, float(cols[1])))
    f_q_values.write("%s target %f\n" % (time, float(cols[2])))

  elif line.startswith("EVAL") and \
       not line.startswith("EVALSTEP") and \
       should_emit("EVAL"):
    cols = line.split(" ")
    if len(cols) == 3:  # format for <= run43
      tag, steps, total_reward = cols
      assert tag == "EVAL"
    elif len(cols) == 4:
      tag, _, steps, total_reward = cols
      assert tag == "EVAL"
    else:
      assert False, line
    assert steps >= 0
    assert total_reward >= 0
    f_eval.write("%s %s %s\n" % (time, steps, total_reward))

  elif line.startswith("NUM_TERMINALS_IN_BATCH") and should_emit("NUM_TERMINALS_IN_BATCH"):
    cols = line.split(" ")
    assert len(cols) == 2
    assert cols[0] == "NUM_TERMINALS_IN_BATCH"
    f_batch_num_terminal.write("%s %f\n" % (time, float(cols[1])))

  elif line.startswith("Q LOSS") and should_emit("Q LOSS"):
    cols = line.split(" ")
    assert len(cols) == 3
    assert cols[0] == "Q"
    assert cols[1] == "LOSS"   # o_O
    f_q_loss.write("%s %f\n" % (time, float(cols[2])))

print "n_parse_errors", n_parse_errors
print freq

