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

import net.hudup.core.alg.ExecutableAlgRemoteWrapper;
import net.hudup.core.data.AttributeList;
import net.hudup.core.data.DataConfig;
import net.hudup.core.logistic.BaseClass;
import net.hudup.core.logistic.Inspector;
import net.hudup.core.logistic.LogUtil;
import net.hudup.core.logistic.xURI;
import net.rem.regression.ui.graph.Graph;

/**
 * The class is a wrapper of remote regression algorithm. This is a trick to use RMI object but not to break the defined programming architecture.
 * In fact, RMI mechanism has some troubles or it it affect negatively good architecture.
 * For usage, an algorithm as REM will has a pair: REM stub (remote regression algorithm) and REM wrapper (normal regression algorithm).
 * The server creates REM stub (remote regression algorithm) and the client creates and uses the REM wrapper as normal regression algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@BaseClass //The annotation is very important which prevent Firer to instantiate the wrapper without referred remote object. This wrapper is not normal algorithm.
public class RMRemoteWrapper extends ExecutableAlgRemoteWrapper implements RM, RMRemote {

	
	/**
	 * Default serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with specified remote regression model.
	 * @param remoteRM specified remote regression model.
	 */
	public RMRemoteWrapper(RMRemote remoteRM) {
		super(remoteRM);
	}

	
	/**
	 * Constructor with specified remote regression model and exclusive mode.
	 * @param remoteRM specified remote regression model.
	 * @param exclusive specified exclusive mode.
	 */
	public RMRemoteWrapper(RMRemote remoteRM, boolean exclusive) {
		super(remoteRM, exclusive);
	}

	
	@Override
	public synchronized Inspector getInspector() {
		return RMAbstract.getInspector(this);
	}


	@Override
	public String[] getBaseRemoteInterfaceNames() throws RemoteException {
		return new String[] {RMRemote.class.getName()};
	}

	
	@Override
	public Object extractResponseValue(Object input) throws RemoteException {
		return ((RMRemote)remoteAlg).extractResponseValue(input);
	}

	
	@Override
	public LargeStatistics getLargeStatistics() throws RemoteException {
		return ((RMRemote)remoteAlg).getLargeStatistics();
	}

	
	@Override
	public double executeByXStatistic(double[] xStatistic) throws RemoteException {
		return ((RMRemote)remoteAlg).executeByXStatistic(xStatistic);
	}

	
	@Override
	public AttributeList getAttributeList() throws RemoteException {
		return ((RMRemote)remoteAlg).getAttributeList();
	}


	@Override
	public VarWrapper extractRegressor(int index) throws RemoteException {
		return ((RMRemote)remoteAlg).extractRegressor(index);
	}

	
	@Override
	public List<VarWrapper> extractRegressors() throws RemoteException {
		return ((RMRemote)remoteAlg).extractRegressors();
	}

	
	@Override
	public List<VarWrapper> extractSingleRegressors() throws RemoteException {
		return ((RMRemote)remoteAlg).extractSingleRegressors();
	}

	
	@Override
	public double extractRegressorValue(Object input, int index) throws RemoteException {
		return ((RMRemote)remoteAlg).extractRegressorValue(input, index);
	}

	
	@Override
	public double[] extractRegressorValues(Object input) throws RemoteException {
		return ((RMRemote)remoteAlg).extractRegressorValues(input);
	}


	@Override
	public List<Double> extractRegressorStatistic(VarWrapper regressor) throws RemoteException {
		return ((RMRemote)remoteAlg).extractRegressorStatistic(regressor);
	}

	
	@Override
	public VarWrapper extractResponse() throws RemoteException {
		return ((RMRemote)remoteAlg).extractResponse();
	}

	
	@Override
	public Object transformResponse(Object z, boolean inverse) throws RemoteException {
		return ((RMRemote)remoteAlg).transformResponse(z, inverse);
	}

	
	@Override
	public Graph createRegressorGraph(VarWrapper regressor) throws RemoteException {
		return ((RMRemote)remoteAlg).createRegressorGraph(regressor);
	}

	
	@Override
	public Graph createResponseGraph() throws RemoteException {
		return ((RMRemote)remoteAlg).createResponseGraph();
	}

	
	@Override
	public Graph createErrorGraph() throws RemoteException {
		return ((RMRemote)remoteAlg).createErrorGraph();
	}

	
	@Override
	public List<Graph> createResponseRalatedGraphs() throws RemoteException {
		return ((RMRemote)remoteAlg).createResponseRalatedGraphs();
	}

	
	@Override
	public double calcVariance() throws RemoteException {
		return ((RMRemote)remoteAlg).calcVariance();
	}

	
	@Override
	public double calcR() throws RemoteException {
		return ((RMRemote)remoteAlg).calcR();
	}

	
	@Override
	public double calcR(double factor, int index) throws RemoteException {
		return ((RMRemote)remoteAlg).calcR(factor, index);
	}


	@Override
	public double[] calcError() throws RemoteException {
		return ((RMRemote)remoteAlg).calcError();
	}

	
	@Override
	public boolean saveLargeStatistics(xURI uri, int decimal) throws RemoteException {
		return ((RMRemote)remoteAlg).saveLargeStatistics(uri, decimal);
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		if (remoteAlg instanceof RM)
			return ((RM)remoteAlg).createDefaultConfig();
		else {
			LogUtil.warn("Wrapper of remote RM algorithm does not support createDefaultConfig()");
			return null;
		}
	}


}
