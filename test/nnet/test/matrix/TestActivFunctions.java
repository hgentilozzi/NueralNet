package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import nnet.matrix.acvt.ActivationFunctionReLU;
import nnet.matrix.acvt.ActivationFunctionSigmoid;
import nnet.matrix.acvt.ActivationFunctionTanh;
import nnet.matrix.net.*;

class TestActivFunctions {
	private ActivationFunctionReLU afReLU = new ActivationFunctionReLU();
	private ActivationFunctionTanh afStep = new ActivationFunctionTanh();
	private ActivationFunctionSigmoid afSigm = new ActivationFunctionSigmoid();

	@Test
	void testReLU() {
		assertTrue(afReLU.f(-1)==0);
		assertTrue(afReLU.f(0)==0);
		assertTrue(afReLU.f(2.6)==2.6);
	}

	@Test
	void testStep() {
		assertTrue(afStep.f(-1)==0);
		assertTrue(afStep.f(0)==0);
		assertTrue(afStep.f(2.6)==1);
	}


	@Test
	void testSigm() {
		assertTrue(afSigm.f(-2)>0 && afSigm.f(-2)<1);
		assertTrue(afSigm.f(0)>0 && afSigm.f(0)<1);
		assertTrue(afSigm.f(2)>0 && afSigm.f(2)<1);
	}

}
