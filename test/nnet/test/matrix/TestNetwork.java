package nnet.test.matrix;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import nnet.matrix.acvt.ActivationFunction;
import nnet.matrix.data.NNetBatchDataArray;
import nnet.matrix.net.NNetParameters;
import nnet.matrix.net.Network;

class TestNetwork {
	
	@BeforeAll
	static void init() {
		NNetParameters.getInstance().setActivationFunction(ActivationFunction.RELU);
		NNetParameters.getInstance().setBatchSize(1);
		NNetParameters.getInstance().setLearningRate(0.001);		
	}

	@Test
	void test() throws Exception {
		double[][] i_data = new double[][] {{0.8}};				
		double[][] o_data = new double[][] {{1.0}};				
		NNetBatchDataArray bdata = new NNetBatchDataArray(i_data, o_data);

		Network n1 = new Network(1,1,1);
		n1.setDebugLevel(1);
		
		n1.train(bdata, 1);

	}

}
