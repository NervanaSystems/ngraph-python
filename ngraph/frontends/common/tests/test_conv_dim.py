

def test_conv_inverts_deconv(input_size, filter_size, padding, dilation, stride):

    conv_output = conv_output_dim(input_size, filter_size, padding, stride, dilation=dilation)
    deconv_output = deconv_output_dim(conv_output, filter_size, padding, stride, dilation=dilation)

    assert deconv_output == input_size, "{} != {}".format(deconv_output, input_size)
