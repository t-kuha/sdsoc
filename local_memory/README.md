### Execution Result

```bash
# First execution
# Note local memory is initialized with 0
root@z7_20:/run/media/mmcblk0p1# ./local_mem.elf
-------------------------
  Uninitialized Value
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  Initializing Local Memory...
    256 255 254 253 252 251 250 249 248 247 246 245 244 243 242 241
  Reading Back Initialized Value...
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    256 255 254 253 252 251 250 249 248 247 246 245 244 243 242 241
-------------------------

# Second execution
# Note local memory maintains the values from last execution
root@z7_20:/run/media/mmcblk0p1# ./local_mem.elf
-------------------------
  Uninitialized Value
    256 255 254 253 252 251 250 249 248 247 246 245 244 243 242 241
  Initializing Local Memory...
    256 255 254 253 252 251 250 249 248 247 246 245 244 243 242 241
  Reading Back Initialized Value...
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    256 255 254 253 252 251 250 249 248 247 246 245 244 243 242 241
-------------------------
```