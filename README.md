# SDSoC Projects

- Tool version: 2018.3

***

## Prereqisite

```shell-session
# Prepare SDx command
$ . <SDx installation directory>/settings64.sh

# Go into the project directory of choice
$ cd <project name>
```

***

## Finished projects

### algorithm_sort

- Odd-Even sort

### binarynet

- Binary CNN
  - __optimized__: Optimized code (Fits in Zybo (Zynq-7010))
  - __templated__: Templated version (Does not fit in Zybo)
- Original source: http://www.cqpub.co.jp/interface/download/2016/9/IF1609F.zip

### embed\_build\_date

- Example of embedding build date/time into HW

### local\_memory

- Writing to & reading from PL's local memory from PS

#### reVISION

- Xilinx reVISION test code
  - __simple\_copy__: Test code for simple copying of data
  - __histogram__: Example of performance improvement by not using xf::calcHist()

### vhls_video_lib

- Using Vivado HLS video library ( hls::***() ) in SDSoC
- __WARNING:__ this code does not work with SDSoC 2017.4 (2017.2 is OK)


- Performance (CPU clock cycle @ 142.86 [MHz]):

  | Image Size [px] | SW              | HW        | Speed-up |
  |:---------------:|----------------:|----------:|---------:|
  | 1920 x 1080     | 169,424,338,952 | 9,812,102 | 17,266   |
  | 256 x 256       |   3,593,393,664 |   289,420 | 12,415   |

***

## Project in progress

### local\_laplacian

- FPGA implementation of local laplacian pyramid
- Original code: https://github.com/psalvaggio/local_laplacian_filters

### local\_laplacian\_fast

- FPGA implementation of fast local laplacian pyramid
- Original code: 

### pynq\_bnn

- SDSoC implementation of BNN-PYNQ
