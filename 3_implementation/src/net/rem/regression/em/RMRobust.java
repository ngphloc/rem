/**
 * REM: REGRESSION MODELS BASED ON EXPECTATION MAXIMIZATION ALGORITHM
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.em;

import java.rmi.RemoteException;
import java.util.List;

import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.alg.NoteAlg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.xURI;
import net.rem.regression.LargeStatistics;
import net.rem.regression.RMAbstract;
import net.rem.regression.VarWrapper;
import net.rem.regression.em.ui.graph.Graph;

/**
 * This class implements the regression model based on expectation maximization algorithm and robust regressors.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RMRobust extends RMAbstract implements DuplicatableAlg, NoteAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * regressive expectation maximization model as internal estimator.
	 */
	protected REMImpl rem = new REMImpl();
	
	
	/**
	 * Default constructor.
	 */
	public RMRobust() {
		this.config = rem.getConfig();
	}

	
	@SuppressWarnings("unchecked")
	@Override
	protected Object learn0() throws RemoteException {
		rem.setup((Fetcher<Profile>)sample);
		
		ExchangedParameter parameter = rem.getExchangedParameter();
		if (parameter == null) return null;
		this.coeffs = parameter.alpha;
		
		return parameter;
	}

	
	@Override
	public synchronized void unsetup() throws RemoteException {
		super.unsetup();
		rem.unsetup();
	}


	@Override
	public LargeStatistics getLargeStatistics() throws RemoteException {
		return rem.getLargeStatistics();
	}

	
	@Override
	public double executeByXStatistic(double[] xStatistic) throws RemoteException {
		return rem.executeByXStatistic(xStatistic);
	}

	
	@Override
	public List<Double> extractRegressorStatistic(VarWrapper regressor) throws RemoteException {
		return rem.extractRegressorStatistic(regressor);
	}

	
	@Override
	public Graph createRegressorGraph(VarWrapper regressor) throws RemoteException {
		return rem.createRegressorGraph(regressor);
	}

	
	@Override
	public Graph createResponseGraph() throws RemoteException {
		return rem.createResponseGraph();
	}

	
	@Override
	public Graph createErrorGraph() throws RemoteException {
		return rem.createErrorGraph();
	}

	
	@Override
	public List<Graph> createResponseRalatedGraphs() throws RemoteException {
		return rem.createResponseRalatedGraphs();
	}

	
	@Override
	public double calcVariance() throws RemoteException {
		return rem.calcVariance();
	}

	
	@Override
	public double calcR() throws RemoteException {
		return rem.calcR();
	}

	
	@Override
	public double[] calcError() throws RemoteException {
		return rem.calcError();
	}

	
	@Override
	public boolean saveLargeStatistics(xURI uri, int decimal) throws RemoteException {
		return rem.saveLargeStatistics(uri, decimal);
	}

	
	@Override
	public String getName() {
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "rem_robust";
	}

	
	@Override
	public void setName(String name) {
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public String note() {
		return rem.note();
	}


	@Override
	public DataConfig createDefaultConfig() {
		return rem != null ? rem.createDefaultConfig() : super.createDefaultConfig();
	}


}
