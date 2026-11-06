# Beta Detector
- Check recommended pull ups on SD cards
- possibly add ferrite bead to prevent switching noise
- check i2c pullups for pico

- check trace width - might break under heat when soldering
- via in pad is bad with test point on C14
- minimise via in pad
- find smaller inductor and make loop smaller
- rotate C1 to make a ground plane and minimise loop (no need to switch layers twice)
- can via fence to minimise noise
- can keep power section far away to minimise noise
- even analog signals can have coupling between close traces (will also affect PWM which has sharp corners)

# Power Board
- could be good to breakout the pgood pins
- document the efuse (delay times and pull up values)

- spread connectors out
- probe vias for pgood
- avoid routing traces under ics

# solar control
- want to have a good idea of what specific data to find
    - define what viable means
    - mppt might not give great data (not straight)
        - instead track IV curve over time
        - measure orientation
        - this is more of an integration test rather than unit test
- blind vias
- very easily to misplug stuff with so many 2-pin PAs
    - possibly change to have panel connection + intensity combined for each panel bank
- check STLink connector
- check trace width
- check layout for power on all boards
    - double check with datasheet
- swap ground and power layer
- have all horizontal traces on one layer, vertical on the other
- double check distance to edge
- check capacitors on STM32

# Termination
- Assume no 5V power for Nichrome
- Change 2x PH to SMD PA
- Add more vias
- Use a plane

# APRS
- check sma connection
    - check impedance (3.336 mm)
    - ensure enough clearance from ground for impedance traces (5x distance between ground plane and signal)
    - might want to be co-planer waveguide
- check the ground return for antenna
- bypass should be for each pin, not in one batch
- check decoupling for radio chip
    - also should be between the ground and power pins
- check routing for pins 9/10
- put all SMDs on one side
- align mounting holes
- change to JST-PA
- check connector separation to the chips
- add testpoints
- put the debug connector on the same side

# OBC
- length match SPI lines
- move FETs to front
- add fuse to power connector
- check trace width for foxhunt
- same for APRS for antenna (GPS)
- change heater to screw terminal
- check clearance for thermocouple connector
- most advice from the board
- make sure mech spaces GPS antenna appropriately
- check connector clearance to each other (pain in the ass to solder)
- check position of SD slot
