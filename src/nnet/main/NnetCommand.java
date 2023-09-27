package nnet.main;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.matrix.NNetMatrix;
import nnet.matrix.acvt.ActivationFunction;
import nnet.matrix.data.*;
import nnet.matrix.net.NNetParameters;
import nnet.matrix.net.Network;

public class NnetCommand  {

	public static void main(String[] args) throws MxlibInvalidMatrixOp {
		
		// Get a training set

		double[][] i_data = new double[][] {{4.9,3.0,1.4,0.2},{6.9,3.1,4.9,1.5},{7.1,3.0,5.9,2.1}};				
		double[][] o_data = new double[][] {{1,0,0},{0,1,0},{0,0,1}};				

		NNetBatchDataArray trdata = new NNetBatchDataArray(i_data, o_data);
		
		
		// Setup run parameters
		NNetParameters.getInstance().setActivationFunction(ActivationFunction.RELU);
		NNetParameters.getInstance().setBatchSize(1);
		NNetParameters.getInstance().setLearningRate(0.01);
		NNetParameters.getInstance().setUseSoftMax(false);
		
		try {
			Network n1 = new Network(4,3,3);
			n1.setDebugLevel(0);
			n1.setBias(1);
			
			n1.train(trdata, 4000);

			double[][] it_data = new double[][] {{5.1,3.5,1.4,0.2},{7.0,3.2,4.7,1.4},{6.3,3.3,6.0,2.5}};				
			double[][] ot_data = new double[][] {{1,0,0},{0,1,0},{0,0,1}};				

			NNetBatchDataArray tstdata = new NNetBatchDataArray(it_data, ot_data);
			
			while (!tstdata.atEof()) {
				NNetBatch b = tstdata.nextBatch(1);
				NNetMatrix results = n1.predict(b.inData());
				
				results.print("Test Results");
				b.expectedResults().print("Expected");
			}
						
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		

	}
}

