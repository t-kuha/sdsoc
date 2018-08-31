### Effect of memory allocation on datamover type & performance

#### Environment

- SDSoC: 2018.2 for Windows

- Target: Zybo Z7-20


***
#### 1. sds_alloc()

- _hw_func_1()_ and _hw_func_2()_ have been merged.

- Performance Estimation

| Estimated Cycle | DSP | BRAM | LUT  | FF    |
|-----------------|-----|------|------|-------|
| 140746          | 4   | 16   | 8729 | 11593 |


- Data Motion Network

| Accelerator | IP Port | Pragmas | Connection                  |
|-------------|---------|---------|-----------------------------|
| hw_func_1_1 | src     |         | ps7_S_AXI_ACP:AXIDMA_SIMPLE | 
|             | dst     |         | hw_func_2_1:src             | 
|             | count   |         | ps7_M_AXI_GP0:AXILITE:0xC   | 
| hw_func_2_1 | src     |         | hw_func_1_1:dst             | 
|             | dst     |         | ps7_S_AXI_ACP:AXIDMA_SIMPLE | 
|             | count   |         | ps7_M_AXI_GP0:AXILITE:0xC   | 


- Accelerator Callsites

| Accelerator | IP Port  | Transfer Size | Paged or Contiguous | Datamover Setup Time | Transfer Time |
|-------------|----------|--------------:|---------------------|---------------------:|--------------:|
| hw_func_1_1 | src      | 16384         | contiguous          | 1112                 | 28524         |
|             | dst      | 16384         | contiguous          |                      |               |
|             | count    | 4             | paged               |  0                   | 13            |
| hw_func_2_1 | src      | 16384         | contiguous          |  1112                | 28524         |
|             | dst      | 16384         | contiguous          |                      |               |
|             | count    | 4             | paged               | 0                    | 13            |



***
#### 2. sds_alloc_non_cacheable()

- Same as 1.


*** 
#### 3. Apply zero_cpy pragma to _tmp_

- _hw_func_1()_ and _hw_func_2()_ have been instantiated separately.


- Performance Estimation - hw_func_1

| Estimated Cycle | DSP | BRAM | LUT  | FF    |
|-----------------|-----|------|------|-------|
| 140018          | 4   | 18   | 9983 | 12630 |

- Performance Estimation - hw_func_2

| Estimated Cycle | DSP | BRAM | LUT  | FF    |
|-----------------|-----|------|------|-------|
| 87902           |     |      |      |       |


- Data Motion Network

| Accelerator | IP Port | Pragmas              | Connection                  |
|-------------|---------|----------------------|-----------------------------|
| hw_func_1_1 | src     |                      | ps7_S_AXI_ACP:AXIDMA_SIMPLE | 
|             | dst     | data_mover:zero_copy | ps7_S_AXI_HP0:AXIMM:0xC     | 
|             | count   |                      | ps7_M_AXI_GP0:AXILITE:0x10  | 
| hw_func_2_1 | src     | data_mover:zero_copy | ps7_S_AXI_HP1:AXIMM:0xC     | 
|             | dst     |                      | ps7_S_AXI_ACP:AXIDMA_SIMPLE | 
|             | count   |                      | ps7_M_AXI_GP0:AXILITE:0x10  | 


- Accelerator Callsites

| Accelerator | IP Port  | Transfer Size | Paged or Contiguous | Datamover Setup Time | Transfer Time |
|-------------|----------|--------------:|---------------------|---------------------:|--------------:|
| hw_func_1_1 | src      | 16384         | contiguous          | 1112                 | 28524         |
|             | dst      | 16384         | contiguous          | 518                  | 29406         |
|             | count    | 4             | paged               |  0                   | 13            |
| hw_func_2_1 | src      | 16384         | contiguous          |  1112                | 28524         |
|             | dst      | 16384         | contiguous          | 518                  | 29406         |
|             | count    | 4             | paged               | 0                    | 13            |