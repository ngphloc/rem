/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.em;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.Collection;

import net.hudup.core.alg.ExecutableAlgAbstract;
import net.hudup.core.alg.MemoryBasedAlg;
import net.hudup.core.alg.MemoryBasedAlgRemote;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.alg.SetupAlgEvent.Type;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Fetcher;

/**
 * <code>AbstractEM</code> is the most abstract class for expectation maximization (EM) algorithm.
 * It implements partially the interface {@link EM}.
 * For convenience, implementation of an EM algorithm should extend this class.
 * 
 * @author Loc Nguyen
 * @version 1.0*
 */
public abstract class EMAbstract extends ExecutableAlgAbstract implements EM, EMRemote, MemoryBasedAlg, MemoryBasedAlgRemote {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Name of maximum iteration.
	 */
	public final static String EM_MAX_ITERATION_FIELD = "em_max_iteration";
	
	
	/**
	 * Name of epsilon field for EM, stored in configuration.
	 */
	public final static String EM_EPSILON_FIELD = "em_epsilon";

	
	/**
	 * Default value for epsilon ratio mode.
	 */
	public final static String EM_EPSILON_RATIO_MODE_FIELD = "em_epsilon_ratio_mode";

	
	/**
	 * Current iteration.
	 */
	protected int currentIteration = 0;
	
	
	/**
	 * Previous parameter.
	 */
	protected Object previousParameter = null;

	
	/**
	 * Current parameter.
	 */
	protected Object currentParameter = null;
	
	
	/**
	 * Current parameter.
	 */
	protected Object estimatedParameter = null;
	
	
	/**
	 * Current statistics.
	 */
	protected Object statistics = null;

	
	/**
	 * Default constructor.
	 */
	public EMAbstract() {
		super();
	}

	
	@Override
	public /*synchronized*/ void setup(Dataset dataset, Object...info) throws RemoteException {
		unsetup();
		this.dataset = dataset;
		if (info != null && info.length > 0 && (info[0] instanceof Fetcher<?> || info[0] instanceof Collection<?>)) {
			this.sample = info[0];
			if (info.length ==  1)
				info = null;
			else
				info = Arrays.copyOfRange(info, 1, info.length);
		}
		else
			this.sample = fetchSample(dataset);
		
		this.estimatedParameter = this.currentParameter = this.previousParameter = this.statistics = null;
		this.currentIteration = 0;
		
		if (info != null)
			learnStart(info);
		else
			learnStart();
		
		SetupAlgEvent evt = new SetupAlgEvent(
				this,
				Type.done,
				this.getName(),
				dataset,
				" (t = " + this.getCurrentIteration() + ") learned models: " + this.getDescription());
		fireSetupEvent(evt);
	}

	
	/**
	 * Initializing parameter at the first iteration of EM process.
	 * @return initialized parameter at the first iteration of EM process.
	 */
	protected abstract Object initializeParameter();
	
	
	/**
	 * Setting the terminated condition for EM.
	 * The usual terminated condition is that the bias between current parameter and estimated parameter is smaller than a positive predefined epsilon.
	 * However the terminated condition is dependent on particular application.
	 * @param estimatedParameter estimated parameter.
	 * @param currentParameter current parameter.
	 * @param previousParameter previous parameter.
	 * @param info additional information.
	 * @return true if the EM algorithm can stop.
	 */
	protected abstract boolean terminatedCondition(Object estimatedParameter, Object currentParameter, Object previousParameter, Object... info);
	
	
	@Override
	public synchronized int getCurrentIteration() {
		return currentIteration;
	}

	
	/**
	 * Setting current iteration.
	 * @param currentIteration current iteration.
	 */
	public synchronized void setCurrentIteration(int currentIteration) {
		this.currentIteration = currentIteration;
	}
	
	
	/**
	 * Getting previous parameter.
	 * @return previous parameter.
	 */
	public synchronized Object getPreviousParameter() {
		return previousParameter;
	}

	
	/**
	 * Setting previous parameter to this regression model. Please use this method carefully.
	 * @param previousParameter previous parameter.
	 */
	public synchronized void setPreviousParameter(Object previousParameter) {
		this.previousParameter = previousParameter;
	}

	
	@Override
	public synchronized Object getCurrentParameter() {
		return currentParameter;
	}


	/**
	 * Setting current parameter to this regression model. Please use this method carefully.
	 * @param currentParameter current parameter.
	 */
	public synchronized void setCurrentParameter(Object currentParameter) {
		this.currentParameter = currentParameter;
	}

	
	@Override
	public synchronized Object getEstimatedParameter() {
		return estimatedParameter;
	}


	/**
	 * Setting estimated parameter to this regression model. Please use this method carefully.
	 * @param estimatedParameter estimated parameter.
	 */
	public synchronized void setEstimatedParameter(Object estimatedParameter) {
		this.estimatedParameter = estimatedParameter;
	}
	
	
	@Override
	public Object getParameter() throws RemoteException {
		return getEstimatedParameter();
	}
	
	
	/**
	 * Notifying initialization in learning process.
	 */
	protected void initializeNotify() {
		
	}

	
	/**
	 * Notifying permutation in learning process.
	 */
	protected void permuteNotify() {
		
	}
	
	
	/**
	 * Notifying finish in learning process.
	 */
	protected void finishNotify() {
		
	}

	
	/**
	 * Getting maximum number of iterations.
	 * @return maximum number of iterations.
	 */
	public int getMaxIteration() {
		DataConfig config = getConfig();
		int maxIteration = 0;
		if (config.containsKey(EM_MAX_ITERATION_FIELD))
			maxIteration = config.getAsInt(EM_MAX_ITERATION_FIELD);
		if (maxIteration <= 0)
			return EM_MAX_ITERATION;
		else
			return maxIteration;
	}
	
	
	@Override
	public synchronized Object getStatistics() throws RemoteException {
		return statistics;
	}


	/**
	 * Setting current statistics.
	 * @param statistics specified statistics.
	 */
	public synchronized void setStatistics(Object statistics) {
		this.statistics = statistics;
	}
	
	
	@Override
	public String[] getBaseRemoteInterfaceNames() throws RemoteException {
		return new String[] {EMRemote.class.getName()};
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		DataConfig config = super.createDefaultConfig();
		config.put(EM_EPSILON_FIELD, EM_EPSILON);
		config.put(EM_EPSILON_RATIO_MODE_FIELD, EM_EPSILON_RATIO_MODE);
		config.put(EM_MAX_ITERATION_FIELD, EM_MAX_ITERATION);
		return config;
	}


}
