# How to build this repo

- SDSoC: 2018.3

## Build zlib

- The output will be produced in ``zlib-1.2.11/_install/``

```shell-session
$ wget http://www.zlib.net/zlib-1.2.11.tar.gz
$ tar xf zlib-1.2.11.tar.gz
$ cd zlib-1.2.11/

# For Zynq-7000
$ export CROSS_PREFIX=arm-linux-gnueabihf-
# For Zynq MPSoC
$ export CROSS_PREFIX=aarch64-linux-gnu-

$ CROSS_PREFIX=${CROSS_PREFIX} ./configure --prefix=$(pwd)/_install
$ make -j$(nproc) install
$ cd ..
```

## Build bnn-fpga

```shell-session
# For Zynq-7000:
$ export CROSS_PREFIX=arm-linux-gnueabihf-
# For Zynq MPSoC:
$ export CROSS_PREFIX=aarch64-linux-gnu-

# CPU-only version:
$ make -C cpp -j$(nproc)

# HW-accelerated version:
$ PLATFORM=<path to patform directory> make -C cpp/accel/sdsoc_build/ -j$(nproc)
```

## Run

- Download parameter & data files from [the original author's Google Drive](https://drive.google.com/drive/folders/1QC2PP209d7mh2aXUJ3j433Ij7aHBNP0I?usp=sharing) & place the dowloaded files into ``data`` & ``params``

- Copy the content of ``data``, ``params``, ``sd_card`` & ``setup.sh`` into SD card

- Boot the board & run

```shell-session
root@sd_blk:~# . ./setup.sh
root@sd_blk:~# ./accel_test_bnn.exe 10000
* WT_WORDS   = 4682
* KH_WORDS   = 64
## Loading input data ##
## Loading parameters ##
## Running BNN for 10000 images
  Pred/Label:    3/ 3   [ OK ]

...

  Pred/Label:    7/ 7   [ OK ]

Errors: 1119 (11.19%)

Total accel runtime =   179.8452 seconds

xl-Conv1            :  10000 calls;  0.983 secs total time
xl-Conv2            :  10000 calls; 19.143 secs total time
xl-Conv3            :  10000 calls; 23.623 secs total time
xl-Conv4            :  20000 calls; 29.117 secs total time
xl-Conv5            :  40000 calls; 31.841 secs total time
xl-Conv6            :  80000 calls; 43.607 secs total time
xl-FC1              : 320000 calls; 27.614 secs total time
xl-FC2              :  40000 calls;  2.903 secs total time
xl-FC3              :  10000 calls;  1.013 secs total time
```
