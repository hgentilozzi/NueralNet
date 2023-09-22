package nnet.main;

import nnet.exception.NNetInvalidMatrixOp;
import nnet.matrix.NNetMatrix;
import nnet.matrix.acvt.ActivationFunction;
import nnet.matrix.data.*;
import nnet.matrix.net.NNetParameters;
import nnet.matrix.net.Network;

public class NnetCommand  {

	public static void main(String[] args) throws NNetInvalidMatrixOp {
		
		// Get a training set
		NNetBatchDataFile bdata = new NNetBatchDataFile("data/TopOrBottomDatabase.csv",3,1);	
		
		
		// Setup run parameters
		NNetParameters.getInstance().setActivationFunction(ActivationFunction.SIGMOID);
		NNetParameters.getInstance().setBatchSize(1);
		NNetParameters.getInstance().setLearningRate(0.001);
		
			
		try {
			Network n1 = new Network(3,2,1);
			n1.setDebugLevel(0);
			//n1.enableBias();
			
			n1.train(bdata, 5000);

			double[][] i_data = new double[][] {{60,80,5}};				
			double[][] o_data = new double[][] {{82}};				
			NNetBatchDataArray tdata = new NNetBatchDataArray(i_data, o_data);
			
			while (!tdata.atEof()) {
				NNetBatch b = tdata.nextBatch(1);
				NNetMatrix results = n1.predict(b.inData());
				
				results.print("Test Results");
				b.expectedResults().print(
						"Expected");
			}

			
			
		} catch (Exception e) {
			System.err.println(e);
		}
		

	}
}

