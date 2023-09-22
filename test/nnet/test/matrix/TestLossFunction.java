package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import nnet.loss.NNetLossFunction;
import nnet.matrix.Matrix;

class TestLossFunction {

	@Test
	void testSquaredError() {
		Matrix e = new Matrix(new double[][] {{5,3}});
		Matrix a = new Matrix(new double[][] {{-4,2}});

		double loss = NNetLossFunction.SQUARED_ERROR.getLoss(a, e);
		assertEquals(81+1,loss);	
	}

	@Test
	void testMeanSquaredError() {
		Matrix e = new Matrix(new double[][] {{5,3}});
		Matrix a = new Matrix(new double[][] {{-4,2}});

		double loss = NNetLossFunction.MEAN_SQRD_ERROR.getLoss(a, e);
		assertEquals(82/2,loss);	
	}


	@Test
	void testMeanAbsoluteErrore() {
		Matrix e = new Matrix(new double[][] {{5,3}});
		Matrix a = new Matrix(new double[][] {{7,2}});

		double loss = NNetLossFunction.ABSOLUTE_ERROR.getLoss(a, e);
		assertEquals(1.5,loss);	
	}


}
