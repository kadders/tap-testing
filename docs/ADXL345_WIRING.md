# ADXL345 wiring and interface (Raspberry Pi)

This project supports the ADXL345 over **I2C** or **SPI** on Raspberry Pi. The same pinout and wiring used by [Klipper’s resonance measurement guide](https://www.klipper3d.org/Measuring_Resonances.html#adxl345) can be used here.

## Why SPI vs I2C

From Klipper’s docs:

> You need to connect ADXL345 to your Raspberry Pi via **SPI**. Note that the **I2C connection, which is suggested by ADXL345 documentation, has too low throughput and will not work** [for high-rate resonance testing].

For **tap testing** at typical rates (e.g. 800–1600 Hz), **I2C can work** when the bus and wiring are good. If you see I2C errors (e.g. “Remote I/O”, “No I2C device”), or want more margin for higher sample rates, use **SPI** and the wiring below.

**Board note:** Some ADXL345 boards are I2C-only (SDO tied to GND). For SPI, the board must support SPI mode (SDO not hardwired for I2C).

---

## SPI wiring (recommended by Klipper)

Use **SPI** and the same pins as Klipper so you can reuse the same cable and layout.

| ADXL345 pin | RPi pin | RPi pin name           |
|-------------|---------|------------------------|
| 3V3 (VCC)   | 01      | 3.3 V DC power         |
| GND         | 06      | Ground                 |
| CS          | 24      | GPIO 8 (SPI0_CE0_N)    |
| SDO         | 21      | GPIO 9 (SPI0_MISO)     |
| SDA         | 19      | GPIO 10 (SPI0_MOSI)    |
| SCL         | 23      | GPIO 11 (SPI0_SCLK)    |

- Enable **SPI** in `raspi-config` → Interface Options → SPI.
- In config, set `adxl345_interface: "spi"` and `spi_cs_pin: 8` (BCM GPIO 8 = CE0).
- **ADXL345 requires SPI Mode 3** (CPOL=1, CPHA=1). To check mode on all SPI devices:
  `python -m tap_testing.check_spi_mode --all`
  To check one device: `python -m tap_testing.check_spi_mode --bus 0 --device 1`
- **Multiple spidev devices:** If you have more than one (e.g. `/dev/spidev0.0`, `/dev/spidev0.1`), set which one to use with `TAP_SPI_BUS` and `TAP_SPI_DEVICE`, or probe to find the ADXL345:
  `python -m tap_testing.verify_spi_accel --probe`

- **DEVID 0x00 / “no ADXL345”:** When probe fails, the script prints a raw DEVID read (tx/rx bytes). If the **first** byte in `rx` is `0xE5`, your board returns data in the first byte—set `TAP_SPI_FIRST_BYTE_DATA=1` and run again. If **all** bytes are `0x00`, MISO is likely floating or no device on that CS: double-check SDO→MISO (GPIO 9), CS→CE0 (Pin 24) or CE1 (Pin 26), and power. Some boards are I2C-only (SDO tied for I2C); try `i2cdetect -y 1` (and other buses) to see if the part appears on I2C.

---

## Verifying that data is coming in

Use these in order to confirm the accelerometer is connected and the values you see are real.

| Step | Command | What it does |
|------|--------|---------------|
| **1. SPI / device** | `python -m tap_testing.verify_spi_accel --probe` | Finds which spidev has ADXL345 (DEVID 0xE5). Run first so you know the chip is seen. |
| **2. Stream + gravity check** | `python -m tap_testing.verify_spi_accel --samples 20 --rate 100` | Streams 20 samples, prints X,Y,Z in g. Pass = non-zero data; also reports mean \|a\| (should be ~1.0 g at rest). Confirms data is live and roughly correct. |
| **3. Live stream** | `python -m tap_testing.check_sensor` | Prints X,Y,Z and magnitude until Ctrl+C. At rest, Z ≈ ±1 g and \|a\| ≈ 1 g if the sensor is level. |
| **4. Record then inspect** | `python -m tap_testing.record_tap` then `python -m tap_testing.inspect_tap_data data/tap_001.csv` | Records one tap to CSV; inspect prints per-axis min/max/std and **Signal: OK (usable)** or **NO SIGNAL**. |
| **5. After a cycle** | `python -m tap_testing.inspect_tap_data data/cycle/<timestamp>` | Same as above for all tap_*.csv and combined.csv in a cycle dir. |

- **At rest:** With the board level, you should see Z ≈ 1 g (or −1 g if Z points down), X and Y near 0, and magnitude ≈ 1 g. If everything is 0 or constant, the data is not coming from the sensor (wiring/SPI/interface).
- **During a tap:** Axes should show a short burst of variation; analysis uses that to find the dominant frequency.

---

## I2C wiring

| ADXL345 pin | RPi pin | RPi pin name   |
|-------------|---------|----------------|
| 3V3 (VCC)   | 01 or 17| 3.3 V          |
| GND         | 06, 09, etc. | Ground    |
| SDA         | 03      | GPIO 2 (SDA1)  |
| SCL         | 05      | GPIO 3 (SCL1)  |

- Enable **I2C** in `raspi-config` → Interface Options → I2C.
- List buses: `i2cdetect -l`. Probe with `i2cdetect -y 1` (and `-y 0`, `-y 20`, `-y 21` if you have multiple buses).
- Default config uses I2C bus 1 and address 0x53. Use `TAP_I2C_BUS=20` (or set `i2c_bus` in config) if the device appears on another bus.

---

## Klipper code reference

Our SPI protocol is aligned with **Klipper’s klippy** implementation (reference directory in this workspace: `klippy/extras/adxl345.py`). That directory is **read-only**; any changes to match Klipper behavior are made in this repo only.

| Item | Klipper (klippy) | tap-testing |
|------|------------------|-------------|
| Read register | `spi_transfer([reg \| 0x80, 0x00])`, use **response[1]** | Same; we also support first-byte-as-data for some boards |
| Write register | `spi_send([reg, val & 0xFF])` | Same (`_write_reg`) |
| SPI mode / speed | Mode 3, 5 MHz | Same |
| DEVID | Check before any write, expect 0xE5 | Same |
| Data path | FIFO + bulk MCU reads for high rate | Poll DATAX0..DATAZ1 (no FIFO) for tap testing |

We use the same register constants (DEVID, POWER_CTL, DATA_FORMAT, DATAX0, MOD_READ, MOD_MULTI) and init order (DEVID then POWER_CTL 0x08). We do not use BW_RATE or FIFO_CTL because we poll at a lower rate.

---

## References

- [Klipper: Measuring Resonances – ADXL345](https://www.klipper3d.org/Measuring_Resonances.html#adxl345) – SPI wiring and RPi connection.
- [Klipper: Wiring](https://www.klipper3d.org/Measuring_Resonances.html#wiring) – Signal integrity (e.g. twisted pairs, pull-ups for I2C).
