/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression;

import java.rmi.RemoteException;
import java.util.List;

import net.hudup.core.alg.ExecutableAlgRemote;
import net.hudup.core.logistic.xURI;
import net.rem.regression.em.ui.graph.Graph;

/**
 * This interface represents remote regression algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface RMRemote extends RMRemoteTask, ExecutableAlgRemote {
    
    
	@Override
	Object extractResponseValue(Object input) throws RemoteException;


	@Override
	LargeStatistics getLargeStatistics() throws RemoteException;

	
	@Override
	double executeByXStatistic(double[] xStatistic) throws RemoteException;

	
	@Override
	VarWrapper extractRegressor(int index) throws RemoteException;

	
	@Override
    List<VarWrapper> extractRegressors() throws RemoteException;
    
    
	@Override
    List<VarWrapper> extractSingleRegressors() throws RemoteException;

    
	@Override
	double extractRegressorValue(Object input, int index) throws RemoteException;

	
	@Override
	double[] extractRegressorValues(Object input) throws RemoteException;

		
	@Override
	List<Double> extractRegressorStatistic(VarWrapper regressor) throws RemoteException;
	
	
	@Override
	VarWrapper extractResponse() throws RemoteException;

	
	@Override
	Object transformResponse(Object z, boolean inverse) throws RemoteException;
	
	
	@Override
    Graph createRegressorGraph(VarWrapper regressor) throws RemoteException;

    
	@Override
    Graph createResponseGraph() throws RemoteException;

    	
	@Override
    Graph createErrorGraph() throws RemoteException;

    	
	@Override
    List<Graph> createResponseRalatedGraphs() throws RemoteException;

    
	@Override
    double calcVariance() throws RemoteException;
    
    
	@Override
    double calcR()throws RemoteException ;
    
    
	@Override
    double[] calcError() throws RemoteException ;
    
    
	@Override
    boolean saveLargeStatistics(xURI uri, int decimal) throws RemoteException;


}
