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

import net.hudup.core.alg.AlgAbstract;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.alg.NoteAlg;
import net.hudup.core.data.DataConfig;
import net.hudup.core.logistic.Inspector;
import net.hudup.core.logistic.xURI;
import net.rem.regression.LargeStatistics;
import net.rem.regression.RMAbstract;
import net.rem.regression.VarWrapper;
import net.rem.regression.em.ui.graph.Graph;

/**
 * This class implements the regression model based on expectation maximization algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class REMInclude extends RMAbstract implements DuplicatableAlg, NoteAlg {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * regressive expectation maximization model as internal estimator.
	 */
	protected REM rem = null;
	
	
	/**
	 * Default constructor.
	 */
	public REMInclude() {
		super();
		rem = createREM();
		
		if (rem != null) {
			config = createDefaultConfig();
			if (rem instanceof AlgAbstract)
				((AlgAbstract)rem).setConfig(config);
			else
				config = rem.getConfig();
		}
	}

	
	/**
	 * Creating regressive expectation maximization (REM) algorithm.
	 * @return regressive expectation maximization (REM) algorithm.
	 */
	protected abstract REM createREM();

	
	@Override
	public synchronized void unsetup() throws RemoteException {
		super.unsetup();
		rem.unsetup();
	}


	@Override
	public synchronized double executeByXStatistic(double[] xStatistic) throws RemoteException {
		return rem.executeByXStatistic(xStatistic);
	}

	
	@Override
	public synchronized Object execute(Object input) throws RemoteException {
		return rem.execute(input);
	}


	@Override
	public synchronized Object getParameter() throws RemoteException {
		return rem.getParameter();
	}


	@Override
	public LargeStatistics getLargeStatistics() throws RemoteException {
		return rem.getLargeStatistics();
	}

	
	@Override
	public String parameterToShownText(Object parameter, Object... info) throws RemoteException {
		return rem.parameterToShownText(parameter, info);
	}


	@Override
	public synchronized String getDescription() throws RemoteException {
		return rem.getDescription();
	}

	
	@Override
	public synchronized Inspector getInspector() {
		return rem.getInspector();
	}


	@Override
	public String note() {
		if (rem instanceof REMAbstract)
			return ((REMAbstract)rem).note();
		else
			return "";
	}


	@Override
	public VarWrapper extractRegressor(int index) throws RemoteException {
		return rem.extractRegressor(index);
	}

	
	@Override
	public List<VarWrapper> extractRegressors() throws RemoteException {
		return rem.extractRegressors();
	}


	@Override
	public List<VarWrapper> extractSingleRegressors() throws RemoteException {
		return rem.extractSingleRegressors();
	}


	@Override
	public double extractRegressorValue(Object input, int index) throws RemoteException {
		return rem.extractRegressorValue(input, index);
	}


	@Override
	public double[] extractRegressorValues(Object input) throws RemoteException {
		return rem.extractRegressorValues(input);
	}


	@Override
	public synchronized List<Double> extractRegressorStatistic(VarWrapper regressor) throws RemoteException {
		return rem.extractRegressorStatistic(regressor);
	}

	
	@Override
	public VarWrapper extractResponse() throws RemoteException {
		return rem.extractResponse();
	}


	@Override
	public synchronized Object extractResponseValue(Object input) throws RemoteException {
		return rem.extractResponseValue(input);
	}

	
	@Override
	public Object transformResponse(Object z, boolean inverse) throws RemoteException {
		return rem.transformResponse(z, inverse);
	}


	@Override
	public synchronized Graph createRegressorGraph(VarWrapper regressor) throws RemoteException {
		return rem.createRegressorGraph(regressor);
	}

	
	@Override
	public synchronized Graph createResponseGraph() throws RemoteException {
		return rem.createResponseGraph();
	}

	
	@Override
	public synchronized Graph createErrorGraph() throws RemoteException {
		return rem.createErrorGraph();
	}

	
	@Override
	public synchronized List<Graph> createResponseRalatedGraphs() throws RemoteException {
		return rem.createResponseRalatedGraphs();
	}

	
	@Override
	public synchronized double calcVariance() throws RemoteException {
		return rem.calcVariance();
	}

	
	@Override
	public synchronized double calcR(double factor) throws RemoteException {
		return rem.calcR(factor);
	}

	
	@Override
	public synchronized double calcR(double factor, int index) throws RemoteException {
		return rem.calcR(factor, index);
	}


	@Override
	public synchronized double[] calcError() throws RemoteException {
		return rem.calcError();
	}

	
	@Override
	public boolean saveLargeStatistics(xURI uri, int decimal) throws RemoteException {
		return rem.saveLargeStatistics(uri, decimal);
	}

	
	@Override
	public void setName(String name) {
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public DataConfig createDefaultConfig() {
		return rem != null ? rem.createDefaultConfig() : super.createDefaultConfig();
	}


}
