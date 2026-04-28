#!/usr/bin/env bash
# lib/powermetrics_probe.sh
# Background sampling of GPU and CPU power via macOS powermetrics.
# Source this from a test script; do not execute directly.
#
# powermetrics requires sudo. The test that uses this library checks
# `sudo -n` availability up front and fails with exit code 3 if not
# available, so the user can decide how to grant it (cached sudo,
# /etc/sudoers entry for the specific binary, etc.).

# Start background powermetrics sampling. Args: output file, duration_s.
# Returns the PID of the background process via the global variable
# POWERMETRICS_PID.
POWERMETRICS_PID=""

powermetrics_start() {
  local outfile="$1"
  local duration="$2"
  # Sample GPU and CPU power at 1-second intervals.
  sudo -n powermetrics \
    --samplers gpu_power,cpu_power \
    -i 1000 \
    -n "$duration" \
    > "$outfile" 2>&1 &
  POWERMETRICS_PID=$!
}

# Stop the background sampler.
#
# We invoke `sudo -n killall powermetrics`, which terminates all
# active powermetrics processes. This requires a NOPASSWD sudoers
# entry scoped to /usr/bin/killall powermetrics:
#
#   echo "$USER ALL=(root) NOPASSWD: /usr/bin/killall powermetrics" \
#     | sudo tee -a /etc/sudoers.d/powermetrics-nopasswd
#
# If that fails (no NOPASSWD for killall, etc.), we fall back to
# waiting for powermetrics to hit its own -n cap. That blocks the
# script for the full sampler duration but does not corrupt data —
# the windowed-mean helpers discard trailing idle samples.
#
# Note: killall ends ALL powermetrics processes the user can affect.
# If you have an unrelated powermetrics running concurrently, this
# stops it too. For our test harness that's fine; tests run serially.
powermetrics_stop() {
  if [[ -z "$POWERMETRICS_PID" ]]; then
    return 0
  fi

  if sudo -n /usr/bin/killall powermetrics 2>/dev/null; then
    # Killed cleanly. Reap our backgrounded sudo wrapper.
    wait "$POWERMETRICS_PID" 2>/dev/null || true
  else
    # Fall back to waiting for powermetrics to exit on its own.
    echo "  powermetrics_stop: killall via sudo not authorised; waiting for sampler to finish" >&2
    wait "$POWERMETRICS_PID" 2>/dev/null || true
  fi
  POWERMETRICS_PID=""
}

# Compute mean GPU power (mW) from a powermetrics output file.
# Returns numeric mean to stdout, or "-1" if no samples found.
powermetrics_mean_gpu_mw() {
  local file="$1"
  awk '
    /GPU Power:/ {
      # Lines look like: "GPU Power: 1234 mW"
      for (i=1; i<=NF; i++) {
        if ($i ~ /^[0-9]+$/) { sum += $i; n++; break }
      }
    }
    END {
      if (n > 0) printf "%.0f\n", sum/n
      else print "-1"
    }
  ' "$file"
}

# Compute mean GPU power over the FIRST N samples (windowed mean).
# Use this when you know how long the workload actually ran for and
# don't want trailing idle samples to dilute the mean. Args: file,
# sample count (typically equal to elapsed seconds, since we sample
# at 1Hz). Returns "-1" if no samples found.
powermetrics_mean_gpu_mw_window() {
  local file="$1"
  local lim="$2"
  awk -v lim="$lim" '
    /GPU Power:/ && n < lim {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^[0-9]+$/) { sum += $i; n++; break }
      }
    }
    END {
      if (n > 0) printf "%.0f\n", sum/n
      else print "-1"
    }
  ' "$file"
}

# Compute mean CPU power (mW) from a powermetrics output file.
powermetrics_mean_cpu_mw() {
  local file="$1"
  awk '
    /^CPU Power:/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^[0-9]+$/) { sum += $i; n++; break }
      }
    }
    END {
      if (n > 0) printf "%.0f\n", sum/n
      else print "-1"
    }
  ' "$file"
}

# Windowed CPU power mean (see _gpu_mw_window for rationale).
powermetrics_mean_cpu_mw_window() {
  local file="$1"
  local lim="$2"
  awk -v lim="$lim" '
    /^CPU Power:/ && n < lim {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^[0-9]+$/) { sum += $i; n++; break }
      }
    }
    END {
      if (n > 0) printf "%.0f\n", sum/n
      else print "-1"
    }
  ' "$file"
}

# Verify powermetrics is invokable without password prompt.
# Tests the actual binary rather than `sudo -n true`, so a NOPASSWD
# entry scoped to /usr/bin/powermetrics is correctly detected.
# Returns 0 if usable, 3 if not.
powermetrics_check_sudo() {
  # One-shot, one-sample probe. Discards output. If this succeeds
  # without prompting, the test can use powermetrics in earnest.
  if sudo -n /usr/bin/powermetrics --samplers gpu_power -i 1000 -n 1 \
       >/dev/null 2>&1; then
    return 0
  fi
  echo "powermetrics requires sudo without password prompt." >&2
  echo "Either:" >&2
  echo "  - run \`sudo -v\` first AND your sudoers permits all binaries, OR" >&2
  echo "  - add a NOPASSWD entry for /usr/bin/powermetrics:" >&2
  echo "      echo \"\$USER ALL=(root) NOPASSWD: /usr/bin/powermetrics\" \\" >&2
  echo "        | sudo tee /etc/sudoers.d/powermetrics-nopasswd" >&2
  echo "      sudo chmod 0440 /etc/sudoers.d/powermetrics-nopasswd" >&2
  return 3
}
