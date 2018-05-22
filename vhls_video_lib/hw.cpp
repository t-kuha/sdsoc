#include "hw.h"

#include "hls_video.h"

#pragma SDS data access_pattern(video_in:SEQUENTIAL, video_out:SEQUENTIAL)
#pragma SDS data copy(video_in[0:rows*cols], video_out[0:rows*cols])
//#pragma SDS data mem_attribute(video_in:PHYSICAL_CONTIGUOUS, video_out:PHYSICAL_CONTIGUOUS)
//#pragma SDS data sys_port(video_in:AFI, video_out:AFI)
void hw_top(data_t* video_in, data_t* video_out, int rows, int cols)
{
    HLS_SIZE_T	_rows = rows;
    HLS_SIZE_T	_cols = cols;

    assert(_rows <= _MAX_ROWS_);
    assert(_cols <= _MAX_COLS_);

    hls::Mat<_MAX_ROWS_, _MAX_COLS_, HLS_8UC3> img_0(_rows, _cols);
    hls::Mat<_MAX_ROWS_, _MAX_COLS_, HLS_8UC3> img_1(_rows, _cols);

    hls::Mat<_MAX_ROWS_, _MAX_COLS_, HLS_8UC1> y_0(_rows, _cols);
    hls::Mat<_MAX_ROWS_, _MAX_COLS_, HLS_8UC1> y_1(_rows, _cols);

#pragma HLS DATAFLOW

    hls::Scalar<HLS_MAT_CN(HLS_8UC3), HLS_TNAME(HLS_8UC3)> px;
    for(HLS_SIZE_T r = 0; r < _rows; r++){
        for(HLS_SIZE_T c = 0; c < _cols; c++){
#pragma HLS PIPELINE
        	data_t d = video_in[r*_cols + c];

        	px.val[0] = (/*video_in[r*_cols + c]*/d >> 16) & 0xFF;
        	px.val[1] = (/*video_in[r*_cols + c]*/d >>  8) & 0xFF;
        	px.val[2] = (/*video_in[r*_cols + c]*/d      ) & 0xFF;

        	img_0 << px;
        }
    }
//    hls::Array2Mat<_MAX_COLS_>(video_in/*, rows*/, img_0);

    // RGB -> YUV ?
    hls::CvtColor<HLS_RGB2GRAY>(img_0, y_0);

    // Only (XORDER, YORDER) = (1, 0) or (0, 1)
    hls::Sobel<1, 0, 3/*, hls::BORDER_REPLICATE*/>(y_0, y_1);

    // YUV -> RGB
    hls::CvtColor<HLS_GRAY2RGB>(y_1, img_1);

    for(HLS_SIZE_T r = 0; r < _rows; r++){
        for(HLS_SIZE_T c = 0; c < _cols; c++){
#pragma HLS PIPELINE
        	img_1 >> px;

        	video_out[r*_cols + c] =
        			(px.val[0] << 16) + (px.val[1] << 8) + px.val[2];
        }
    }
//    hls::Mat2Array<_MAX_COLS_>(img_1, video_out/*, rows*/);
}
