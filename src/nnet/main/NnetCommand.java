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
		//NNetBatchDataFile bdata = new NNetBatchDataFile("data/OneRowBatchDB.csv",3,1);	
		
		double[][] i_data = new double[][] {{0.9},{0.7},{0.4}};				
		double[][] o_data = new double[][] {{1.0},{1.0},{0.}};				
		NNetBatchDataArray bdata = new NNetBatchDataArray(i_data, o_data);

		
		// Setup run paramters
		NNetParameters.getInstance().setActivationFunction(ActivationFunction.SIGMOID);
		NNetParameters.getInstance().setBatchSize(1);
		NNetParameters.getInstance().setLearningRate(0.001);
		
			
		try {
			Network n1 = new Network(1,1);
			n1.setDebugLevel(2);
			
			n1.train(bdata, 2);
			
			System.out.println(n1.getStats().toString());
						
			bdata.reset();
			NNetBatch b = bdata.nextBatch(1);
			
			NNetMatrix results = n1.predict(b.inData());
			
			results.print("Test Results");
			b.expectedResults().print(
					"Expected");
			
			
		} catch (Exception e) {
			System.err.println(e);
		}
		

	}
}

