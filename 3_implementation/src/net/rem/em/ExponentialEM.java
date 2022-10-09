/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.em;

import java.io.Serializable;
import java.rmi.RemoteException;

import net.hudup.core.alg.SetupAlgEvent.Type;
import net.hudup.core.logistic.LogUtil;

/**
 * This abstract class model a expectation maximization (EM) algorithm for exponential family.
 * In other words, probabilistic distributions in this class belongs to exponential family.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ExponentialEM extends EMAbstract {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public ExponentialEM() {
		super();
	}

	
	/**
	 * This method implement expectation step (E-step) of EM.
	 * @param currentParameter current parameter.
	 * @param info additional information.
	 * @return sufficient statistic given current parameter.
	 * @throws RemoteException if any error raises
	 */
	protected abstract Object expectation(Object currentParameter, Object...info) throws RemoteException;
	
	
	/**
	 * This method implement maximization step (M-step) of EM.
	 * @param currentStatistic current sufficient statistic.
	 * @param info additional information.
	 * @return estimated parameter given current sufficient statistic.
	 * @throws RemoteException if any error raises
	 */
	protected abstract Object maximization(Object currentStatistic, Object...info) throws RemoteException;
	
	
	/*
	 * In the this version, the learn method is not marked synchronized because it is called by setup method.
	 */
	@Override
	public Object learnStart(Object...info) throws RemoteException {
		if (isLearnStarted()) return null;

		learnStarted = true;

		this.estimatedParameter = this.currentParameter = this.previousParameter = this.statistics = null;
		this.currentIteration = 0;
		this.estimatedParameter = this.currentParameter = initializeParameter();
		initializeNotify();
		if (this.estimatedParameter == null) {
			synchronized (this) {
				learnStarted = false;
				learnPaused = false;

				finishNotify();
				
				notifyAll();
				return null;
			}
		}
		
		this.currentIteration = 1;
		int maxIteration = getMaxIteration();
		while (learnStarted && (maxIteration <= 0 || this.currentIteration < maxIteration)) {
			Object tempStatistics = expectation(this.currentParameter);
			if (tempStatistics == null)
				break;
			
			this.statistics = tempStatistics;
			this.estimatedParameter = maximization(this.statistics);
			if (this.estimatedParameter == null)
				break;
			
			//Firing setup doing event
			try {
				fireSetupEvent(new EMLearningEvent(this, Type.doing, this.dataset,
					this.currentIteration, maxIteration, (Serializable)this.statistics,
					(Serializable)this.currentParameter, (Serializable)this.estimatedParameter));
			}
			catch (Throwable e) {LogUtil.trace(e);}
			
			boolean terminated = terminatedCondition(this.estimatedParameter, this.currentParameter, this.previousParameter);
			if (terminated)
				break;
			else {
				this.previousParameter = this.currentParameter;
				this.currentParameter = this.estimatedParameter;
				this.currentIteration++;
				permuteNotify();
			}
			
			synchronized (this) {
				while (learnPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {LogUtil.trace(e);}
				}
			}
			
		} //End while
		
		if (this.estimatedParameter != null)
			this.currentParameter = this.estimatedParameter;
		else if (this.currentParameter != null)
			this.estimatedParameter = this.currentParameter;
		
		synchronized (this) {
			learnStarted = false;
			learnPaused = false;

			//Firing setup done event
			try {
				fireSetupEvent(new EMLearningEvent(this, Type.done, this.dataset,
					this.currentIteration, this.currentIteration, (Serializable)this.statistics,
					(Serializable)this.currentParameter, (Serializable)this.estimatedParameter));
			}
			catch (Throwable e) {LogUtil.trace(e);}
	
			finishNotify();
			
			notifyAll();
		}
		
		return this.estimatedParameter;
	}


}
