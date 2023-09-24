package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.loss.NNetLossFunction;
import nnet.matrix.NNetMatrix;

class TestLossFunction {

	@Test
	void testSquaredError() throws MxlibInvalidMatrixOp {
		NNetMatrix e = new NNetMatrix(new double[][] {{5,3}});
		NNetMatrix a = new NNetMatrix(new double[][] {{-4,2}});

		double loss = NNetLossFunction.SQUARED_ERROR.getLoss(a, e);
		assertEquals(81+1,loss);	
	}

	@Test
	void testMeanSquaredError() throws MxlibInvalidMatrixOp {
		NNetMatrix e = new NNetMatrix(new double[][] {{5,3}});
		NNetMatrix a = new NNetMatrix(new double[][] {{-4,2}});

		double loss = NNetLossFunction.MEAN_SQRD_ERROR.getLoss(a, e);
		assertEquals(82/2,loss);	
	}


	@Test
	void testMeanAbsoluteErrore() throws MxlibInvalidMatrixOp {
		NNetMatrix e = new NNetMatrix(new double[][] {{5,3}});
		NNetMatrix a = new NNetMatrix(new double[][] {{7,2}});

		double loss = NNetLossFunction.ABSOLUTE_ERROR.getLoss(a, e);
		assertEquals(1.5,loss);	
	}


}
