zbc: ./model_components/utils/utils.cpp \
	./model_components/model/SEML/imp/seml.cpp \
	./model_components/model/SEML/imp/seml_v2.cpp \
	./model_components/layers/impl/pw_conv_v1_2.cpp \
	./model_components/layers/impl/pw_conv_v2.cpp \
	./client/prepare_weights_and_input.cpp \
	./client/prepare_weights_and_input_v2.cpp \
	./model_components/layers/impl/pooling.cpp \
	./model_components/layers/impl/norm_act.cpp \
	./client/hls_only_main_file.cpp \
	./model_components/layers/impl/conv_utils.cpp \
	./model_components/layers/impl/dw_conv_v1_6.cpp \
	./model_components/layers/impl/dw_conv_v2.cpp \
	./model_components/layers/impl/conv.cpp \
	./model_components/model/fused_bottlenecks/bottleneck_kernels.cpp \
	./model_components/model/fused_bottlenecks/bottleneck_0.cpp \
	./model_components/model/fused_bottlenecks/bottleneck_1.cpp \
	./model_components/model/fused_bottlenecks/bottleneck_2.cpp \
	./model_components/model/fused_bottlenecks/bottlenecks_chain.cpp \
	./model_components/model/fused_bottlenecks/bottlenecks_chain_0_1_2.cpp \
	./model_components/model/pipelined_engines/pipelined_engines.cpp \
	./model_components/model/pipelined_engines/pipeline_main.cpp \
	./tests/test_utils.cpp \
	./tests/main_tester.cpp
	g++ -o main_tester ./model_components/utils/utils.cpp \
	./model_components/model/SEML/imp/seml.cpp \
	./model_components/model/SEML/imp/seml_v2.cpp \
	./model_components/layers/impl/pw_conv_v1_2.cpp \
	./model_components/layers/impl/pw_conv_v2.cpp \
	./client/prepare_weights_and_input.cpp \
	./client/prepare_weights_and_input_v2.cpp \
	./model_components/layers/impl/pooling.cpp \
	./model_components/layers/impl/norm_act.cpp \
	./client/hls_only_main_file.cpp \
	./model_components/layers/impl/conv_utils.cpp \
	./model_components/layers/impl/dw_conv_v1_6.cpp \
	./model_components/layers/impl/dw_conv_v2.cpp \
	./model_components/layers/impl/conv.cpp \
	./model_components/model/fused_bottlenecks/bottleneck_kernels.cpp \
	./model_components/model/fused_bottlenecks/bottleneck_0.cpp \
	./model_components/model/fused_bottlenecks/bottleneck_1.cpp \
	./model_components/model/fused_bottlenecks/bottleneck_2.cpp \
	./model_components/model/fused_bottlenecks/bottlenecks_chain.cpp \
	./model_components/model/fused_bottlenecks/bottlenecks_chain_0_1_2.cpp \
	./model_components/model/pipelined_engines/pipelined_engines.cpp \
	./model_components/model/pipelined_engines/pipeline_main.cpp \
	./tests/test_utils.cpp \
	./tests/main_tester.cpp
