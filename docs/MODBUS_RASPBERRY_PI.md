# Modbus polling on Raspberry Pi (homing GUI)

The homing calibration GUI can optionally log **Modbus** data alongside the ADXL345 stream when you pass **`--modbus`**. This project assumes **RS-485 Modbus RTU** to the spindle VFD (e.g. **H100**); use **TCP** only if you have an Ethernet gateway.

## H100 VFD (our primary target)

See **[MODBUS_H100_VFD.md](MODBUS_H100_VFD.md)** for register map (holding **5**, **11**, **513**; input **0–1**; coils **73–75**), FluidNC references, and the CNC Zone thread on manual RS-485 commands.

Quick start:

```bash
export TAP_MODBUS_PROFILE=h100
export TAP_MODBUS_SERIAL_PORT=/dev/ttyUSB0
python -m tap_testing.homing_gui --modbus
```

Or:

```bash
python -m tap_testing.homing_gui --modbus --modbus-profile h100 --modbus-serial-port /dev/ttyUSB0
```

## What gets recorded

During **Start recording** → **Stop recording**, the app:

1. Writes **`homing.csv`** — accelerometer (unchanged).
2. Writes **`modbus.csv`** — one row per poll with:
   - **`t_s`** — host timestamp (seconds, wall clock).
   - **`hr_0` … `hr_{N-1}`** — holding registers (unsigned 16-bit).
   - **`ir_0` … `ir_{N-1}`** — input registers.
   - **`di_*`**, **`co_*`** — discrete inputs / coils when enabled.

Comment lines at the top of **`modbus.csv`** include **`transport`**, **`unit_id`**, optional **`profile`** (e.g. `h100`), and serial or TCP parameters.

If the Modbus client cannot connect, you still get ADXL data; the run directory may contain **`modbus.csv.error.txt`**.

## Install on Raspberry Pi OS

```bash
cd /path/to/tap-testing
source venv/bin/activate   # if you use a venv
pip install pymodbus
```

RTU uses **pyserial** (dependency of pymodbus for serial clients).

## RTU (RS-485 USB adapter on the Pi) — default path

Connect a **USB–RS485** adapter; the Pi usually shows **`/dev/ttyUSB0`** (check `dmesg` / `ls -l /dev/serial/by-id`).

1. Match **baud rate, parity, stop bits** to the VFD (**19200 8N1** is common on H100 with grblHAL; some setups use **9600**).
2. Wire **A/B** (or **D+/D−**) per the adapter and drive manual; use **termination** / **ground** as required.
3. Example:

```bash
export TAP_MODBUS_TRANSPORT=rtu
export TAP_MODBUS_SERIAL_PORT=/dev/ttyUSB0
export TAP_MODBUS_BAUDRATE=19200
export TAP_MODBUS_UNIT=1
export TAP_MODBUS_POLL_HZ=10
python -m tap_testing.homing_gui --modbus
```

Windows: `TAP_MODBUS_SERIAL_PORT=COM5`.

## TCP (optional)

Use when a PLC or gateway exposes Modbus TCP:

```bash
export TAP_MODBUS_TRANSPORT=tcp
export TAP_MODBUS_HOST=192.168.1.50
export TAP_MODBUS_PORT=502
export TAP_MODBUS_UNIT=1
python -m tap_testing.homing_gui --modbus
```

## Environment reference (`TAP_MODBUS_*`)

| Variable | Meaning | Default (no env) |
|----------|---------|------------------|
| `TAP_MODBUS_PROFILE` | Device preset: **`h100`** | (none) |
| `TAP_MODBUS_TRANSPORT` | `rtu` or `tcp` | **`rtu`** |
| `TAP_MODBUS_HOST` | TCP host | `127.0.0.1` |
| `TAP_MODBUS_PORT` | TCP port | `502` |
| `TAP_MODBUS_UNIT` | Slave / device id | `1` |
| `TAP_MODBUS_SERIAL_PORT` | Serial device (RTU) | `/dev/ttyUSB0` |
| `TAP_MODBUS_BAUDRATE` | Baud (RTU) | **`19200`** |
| `TAP_MODBUS_PARITY` | `N`, `E`, `O`, … | `N` |
| `TAP_MODBUS_STOPBITS` | `1` or `2` | `1` |
| `TAP_MODBUS_BYTESIZE` | Usually `8` | `8` |
| `TAP_MODBUS_POLL_HZ` | Polls per second | `10` |
| `TAP_MODBUS_MAX_HOLDING` | Max holding index (exclusive) | `400` (H100 profile raises to ≥ **520**) |
| `TAP_MODBUS_MAX_INPUT` | Max input register index (exclusive) | `400` (H100 profile raises to ≥ **8**) |
| `TAP_MODBUS_CHUNK` | Registers per read (≤125) | `100` |
| `TAP_MODBUS_MAX_DISCRETE` | Max discrete input index | `128` |
| `TAP_MODBUS_MAX_COILS` | Max coil index | `128` |
| `TAP_MODBUS_NO_DISCRETE` | `1` = skip discrete reads | off |
| `TAP_MODBUS_NO_COILS` | `1` = skip coil reads | off |

**Order:** `TAP_MODBUS_PROFILE=h100` is applied first, then other variables **override** it (so you can set `TAP_MODBUS_BAUDRATE=9600` after the preset).

## Live GUI

With **`--modbus`**, the window adds a **Modbus (live)** tab: traces for **holding and input registers 0–5** (raw). With **`h100` profile**, labels reference the H100 map; **`modbus.csv`** includes **hr_513** and other configured addresses.

## Security note

Modbus TCP is usually **unauthenticated**. Use only on a trusted LAN.

## See also

- `tap_testing/modbus_logging.py` — polling, **`apply_modbus_profile`**, CSV layout.
- `tap_testing/homing_gui.py` — **`--modbus`**, **`--modbus-profile h100`**.
