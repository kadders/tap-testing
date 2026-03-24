# H100 VFD — Modbus RTU (RS-485)

This project targets the **H100** spindle inverter over **Modbus RTU** on RS-485 (USB adapter on a Raspberry Pi or a PC COM port). Community discussion of manual Modbus frames appears in the CNC Zone thread [Modbus RS485 manual commands H100 VFD](https://www.cnczone.com/forums/spindles-vfd/436444-modbus-rs485-manual-commands-h100-vfd.html) (including [post #2593591](https://www.cnczone.com/forums/spindles-vfd/436444-modbus-rs485-manual-commands-h100-vfd.html#post2593591)); the register layout below matches what **FluidNC** documents and implements in source (authoritative for this repo).

## References (FluidNC)

- [H100Protocol.md](https://github.com/bdring/FluidNC/blob/main/FluidNC/src/Spindles/VFD/H100Protocol.md) — tested PDU examples.
- [H100.cpp](https://github.com/bdring/FluidNC/blob/main/FluidNC/src/Spindles/VFD/H100.cpp) — ModbusVFD command strings (hex PDU).

## Serial settings

| Parameter   | Typical value | Notes |
|------------|---------------|--------|
| Baud rate  | **19200**     | grblHAL default for H100; some setups use **9600** — match the VFD menu. |
| Data bits  | 8             | |
| Parity     | None (N)      | |
| Stop bits  | 1             | 8N1 |
| Unit / slave id | **1**   | Change on the drive if needed; set `TAP_MODBUS_UNIT`. |

If the drive and software disagree on baud rate, you get timeouts or CRC errors.

## Addressing (PDU / 0-based)

Our CSV columns **`hr_N`**, **`ir_N`**, **`co_N`** use **0-based** Modbus PDU addresses (same as pymodbus `address=`). Manuals that say “4x0001” are often **1-based** holding: subtract 1 for pymodbus.

## Function codes used on H100 (summary)

| Function | Code | Purpose (H100) |
|----------|------|----------------|
| Read coils | 01 | — |
| Read discrete inputs | 02 | — |
| Read holding registers | 03 | Min/max frequency parameters (e.g. **5**, **11**). |
| Read input registers | 04 | Output / set frequency (see **ir_0**, **ir_1**). |
| Write single coil | 05 | Run: **forward**, **reverse**, **stop** (see coils below). |
| Write single holding | 06 | Command frequency at holding **513** (0x0201). |

## Registers and coils (logged by tap-testing)

### Input registers (FC 04) — `ir_*`

FluidNC reads **start address 0, count 2** for speed feedback:

- **`ir_0`**, **`ir_1`** — related to **output frequency** and **set frequency** (see FluidNC `H100Protocol.md`: first pair / second pair in the 4-byte response). FluidNC scales with `rpm*60/10` style handling in `H100.cpp` (frequency ↔ RPM depends on pole count / nameplate; treat logged values as **raw u16** until you match your manual).

Example from `H100Protocol.md` (illustrative): reading two registers returns data interpreted as running vs set frequency.

### Holding registers (FC 03) — `hr_*`

From `H100Protocol.md` (FC 03 reads):

- **`hr_5`** (address **0x0005**) — **max frequency** (1 register).
- **`hr_11`** (address **0x000B**) — **min frequency** (1 register).

Command / setpoint frequency write (FC 06) uses holding address **0x0201** = **513** decimal:

- **`hr_513`** — written as **frequency × 10** in the worked example: value **0x07D0 (2000)** → **200.0 Hz** (`H100Protocol.md`). When **reading** is supported by your firmware, the same cell may reflect the commanded value.

With **`TAP_MODBUS_PROFILE=h100`** (or `--modbus-profile h100`), the logger expands **`max_holding` to at least 520** so **hr_513** appears in **`modbus.csv`**.

### Coils (FC 05) — `co_*`

FluidNC / manual style coil writes (read-back support varies by firmware):

| Coil address (PDU) | Role |
|--------------------|------|
| **73** (0x0049)    | Forward run (`FF 00` on). |
| **74** (0x004A)    | Reverse run. |
| **75** (0x004B)    | Stop. |

Our poller **reads** coils with FC 01 when enabled; illegal reads are skipped per chunk.

## Enable H100 profile in tap-testing

```bash
export TAP_MODBUS_PROFILE=h100
export TAP_MODBUS_SERIAL_PORT=/dev/ttyUSB0
# Optional if your VFD uses 9600:
# export TAP_MODBUS_BAUDRATE=9600
export TAP_MODBUS_UNIT=1
python -m tap_testing.homing_gui --modbus
```

Or:

```bash
python -m tap_testing.homing_gui --modbus --modbus-profile h100 --modbus-serial-port /dev/ttyUSB0
```

Environment variables still **override** the preset (e.g. `TAP_MODBUS_BAUDRATE` after the profile is applied).

## Related

- [MODBUS_RASPBERRY_PI.md](MODBUS_RASPBERRY_PI.md) — Pi wiring, env reference, TCP vs RTU.
- [grblHAL issue #762](https://github.com/grblHAL/core/issues/762) — H100 RTU notes (19200, unit id 1, RS-485 direction on some boards).
