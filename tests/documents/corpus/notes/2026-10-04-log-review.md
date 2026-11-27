# Log review: 3 October

## What the log shows
The service log for 3 October has a burst of the same pressure sensor timeout error on bench 1
within one minute in the afternoon, then a code 5 stop for pump 2. Earlier in the day there is a
stack trace from the logging service itself, unrelated to the pumps.

## Cause
The sensor cable for pump 2 was loose at its plug. Each time the bench vibrated, the reading dropped
out and the control unit counted a timeout.

## Fix
Plug reseated and the cable tied to the bench frame. No more timeouts since.

## Follow up
- [ ] tie down the sensor cables on all six pumps
- [x] add the loose cable case to the bench checklist
