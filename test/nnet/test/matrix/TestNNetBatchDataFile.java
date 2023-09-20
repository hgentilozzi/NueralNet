package nnet.test.matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import nnet.matrix.data.*;

class TestNNetBatchDataFile {

	@Test
	void testNextBatchValidate() {
		double[][] i_data = new double[][] {{0.9,0.8},{0.7,0.6},{0.4,0.3}};				
		double[][] o_data = new double[][] {{0.3},{0.3},{0.3}};				
		NNetBatchDataArray bdata = new NNetBatchDataArray(3, i_data, o_data);
		
		assertTrue(bdata.validate());

		i_data = new double[][] {{0.9,0.8},{0.7,0.6},{0.4,0.3}};				
		o_data = new double[][] {{0.3},{0.3}};				
		bdata = new NNetBatchDataArray(3, i_data, o_data);
		
		assertFalse(bdata.validate());
		assertEquals("Input data  and expected result array lengths to not match",bdata.getError());		
	}
	
	@Test
	void testNextBatch1() {
		double[][] i_data = new double[][] {{0.9},{0.8},{0.7},{0.6},{0.4},{0.3},{0.2}};				
		double[][] o_data = new double[][] {{0.3},{0.3},{0.3},{0.3},{0.9},{0.9},{0.9}};				
		NNetBatchDataArray bdata = new NNetBatchDataArray(4, i_data, o_data);
		
		assertTrue(bdata.validate());
		assertFalse(bdata.atEof());
		NNetBatch b = bdata.nextBatch();
		assertEquals(4, b.batchSize());
		assertEquals(4, b.inData().getRows());
		assertEquals(1, b.inData().getCols());
		assertEquals(4, b.expectedResults().getRows());
		assertEquals(1, b.expectedResults().getCols());

		assertFalse(bdata.atEof());
		b = bdata.nextBatch();
		assertEquals(3, b.batchSize());
		assertEquals(3, b.inData().getRows());
		assertEquals(1, b.inData().getCols());
		assertEquals(3, b.expectedResults().getRows());
		assertEquals(1, b.expectedResults().getCols());

		assertTrue(bdata.atEof());

		
	}


	@Test
	void testNextBatch2() {
		double[][] i_data = new double[][] {{0.9,0.8},{0.7,0.6},{0.4,0.3}};				
		double[][] o_data = new double[][] {{0.3},{0.3},{0.3}};				
		NNetBatchDataArray bdata = new NNetBatchDataArray(3, i_data, o_data);
		
		assertTrue(bdata.validate());
		assertFalse(bdata.atEof());
		NNetBatch b = bdata.nextBatch();
		assertEquals(3, b.batchSize());
		assertEquals(3, b.inData().getRows());
		assertEquals(2, b.inData().getCols());
		assertEquals(1, b.expectedResults().getCols());
		assertTrue(bdata.atEof());
		
		// Test reset
		bdata.reset();
		assertTrue(bdata.validate());
		assertFalse(bdata.atEof());
		b = bdata.nextBatch();
		assertEquals(3, b.batchSize());
		assertEquals(3, b.inData().getRows());
		assertEquals(2, b.inData().getCols());
		assertEquals(1, b.expectedResults().getCols());
		assertTrue(bdata.atEof());
	
	}

}
