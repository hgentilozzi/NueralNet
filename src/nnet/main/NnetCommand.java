package nnet.main;

import nnet.exception.*;
import nnet.matrix.NNetMatrix;
import nnet.matrix.acvt.*;
import nnet.matrix.data.NNetBatch;
import nnet.matrix.data.NNetBatchDataArray;
import nnet.matrix.net.*;

public class NnetCommand  {

	public static void main(String[] args) throws NNetInvalidMatrixOp {
		double[][] i_data = new double[][] {{0.9},{0.8},{0.7},{0.6},{0.4},{0.3},{0.2},{0.1},};				
		double[][] o_data = new double[][] {{0.3},{0.3},{0.3},{0.3},{0.9},{0.9},{0.9},{0.9},};				
		NNetBatchDataArray bdata = new NNetBatchDataArray(4, i_data, o_data);
		
		try {
			Network n1 = new Network(new ActivationFunctionReLU(),1,2,1);
			n1.setDebugLevel(0);
			
			n1.train(bdata, 1000);
			
			//System.out.println("Training stats: " + n1.getStats().toString());
			
			bdata.reset();
			NNetBatch b = bdata.nextBatch();
			
			NNetMatrix results = n1.predict(b.inData());
			
			results.print("Test Results");
			b.expectedResults().print("Expected");
			
			
		} catch (NNetInvalidNetwork e) {
			System.err.println(e);
			return;
		}
		
		

	}
}
