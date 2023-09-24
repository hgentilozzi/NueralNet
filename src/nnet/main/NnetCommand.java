package nnet.main;

import java.util.List;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.matrix.NNetMatrix;
import nnet.matrix.acvt.ActivationFunction;
import nnet.matrix.data.*;
import nnet.matrix.net.NNetParameters;
import nnet.matrix.net.Network;

public class NnetCommand  {

	public static void main(String[] args) throws MxlibInvalidMatrixOp {
		
		// Get a training set
		NNetBatchDataFile bdata = new NNetBatchDataFile("data/iris/iris.data",4,1);	
		
		
		// Setup run parameters
		NNetParameters.getInstance().setActivationFunction(ActivationFunction.RELU);
		NNetParameters.getInstance().setBatchSize(4);
		NNetParameters.getInstance().setLearningRate(0.01);
		NNetParameters.getInstance().setUseSoftMax(false);
		
		try {
			Network n1 = new Network(4,3,1);
			n1.setDebugLevel(0);
			n1.enableBias();
			
			n1.train(bdata, 500);

			double[][] i_data = new double[][] {{5.1,3.5,1.4,0.2},{7.0,3.2,4.7,1.4},{6.3,3.3,6.0,2.5,3}};				
			double[][] o_data = new double[][] {{1},{2},{3}};				

			NNetBatchDataArray tdata = new NNetBatchDataArray(i_data, o_data);
			
			while (!tdata.atEof()) {
				NNetBatch b = tdata.nextBatch(1);
				NNetMatrix results = n1.predict(b.inData());
				
				results.print("Test Results");
				b.expectedResults().print(
						"Expected");
			}
			
//			List<Double> lossArray = n1.getStats().getLossArray();
//			for (int i=0;i<lossArray.size();i++) {
//				System.out.println(i + "," + String.format("%,.8f",lossArray.get(i)));
//				
//			}

			
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		

	}
}

