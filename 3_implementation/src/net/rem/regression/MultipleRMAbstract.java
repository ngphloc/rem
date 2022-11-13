/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;

import net.hudup.core.Util;
import net.hudup.core.alg.ExecutableAlgAbstract;
import net.hudup.core.alg.MemoryBasedAlg;
import net.hudup.core.alg.MemoryBasedAlgRemote;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.alg.SetupAlgEvent.Type;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Fetcher;
import net.hudup.core.data.Profile;

/**
 * This is the most abstract class for multiple regression model. It implements partially the interface {@link RM}.
 * 
 * @author Loc Nguyen
 * @version 1.0*
 */
public abstract class MultipleRMAbstract extends ExecutableAlgAbstract implements MultipleRM, MultipleRMRemote, MemoryBasedAlg, MemoryBasedAlgRemote {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of internal regression model as parameter.
	 */
	protected List<RM> regressions = Util.newList(); 
	
	
	@Override
	protected Object fetchSample(Dataset dataset) {
		return dataset != null ? dataset.fetchSample() : null;
	}

	
	@Override
	public void setup(Dataset dataset, Object... info) throws RemoteException {
		List<Object> additionalInfo = Util.newList();
		List<RM> regressions = Util.newList();
		for (Object el : info) {
			if (el instanceof RM)
				regressions.add((RM)el);
			else
				additionalInfo.add(el);
		}
		this.setup(dataset, additionalInfo.toArray(), regressions.toArray(new RM[] {}));
	}


	@Override
	public void setup(Fetcher<Profile> sample, Object... info) throws RemoteException {
		List<Object> additionalInfo = Util.newList();
		List<RM> regressions = Util.newList();
		for (Object el : info) {
			if (el instanceof RM)
				regressions.add((RM)el);
			else
				additionalInfo.add(el);
		}

		this.setup(sample, additionalInfo.toArray(), regressions.toArray(new RM[] {}));
	}


	/**
	 * Setting up this multiple regression algorithm based on specified dataset and many partial regressions.
	 * In the this version, the setup method is not marked synchronized because it calls learnStart method.
	 * @param dataset specified dataset.
	 * @param info additional parameters to set up this algorithm.
	 * @param regressions many specified partial regressions. 
	 * @throws RemoteException if any error raises.
	 */
	@SuppressWarnings("unchecked")
	public /*synchronized*/ void setup(Dataset dataset, Object[] info, RM...regressions) throws RemoteException {
		unsetup();
		for (RM regression : regressions) {
			this.regressions.add(regression);
		}
		if (this.regressions.size() == 0)
			return;

		this.dataset = dataset;
		if (info != null && info.length > 0 && (info[0] instanceof Fetcher<?>))
			this.sample = (Fetcher<Profile>)info[0];
		else
			this.sample = dataset.fetchSample();
		
		List<String> cfgIndicesList = Indices.splitIndices(this.getConfig().getAsString(RM_INDICES_FIELD));
		for (int i = 0; i < Math.min(cfgIndicesList.size(), this.regressions.size()); i++) {
			this.regressions.get(i).getConfig().put(RM_INDICES_FIELD, cfgIndicesList.get(i));
		}
		
		learnStart();
		
		SetupAlgEvent evt = new SetupAlgEvent(
			this,
			Type.done,
			this.getName(),
			dataset,
			"Learned models: " + this.getDescription());
		fireSetupEvent(evt);
	}

	
	/**
	 * Setting up this multiple regression algorithm based on specified sample and many partial regressions.
	 * @param sample specified sample.
	 * @param info additional parameters to set up this algorithm.
	 * @param regressions many specified partial regressions. 
	 * @throws RemoteException if any error raises.
	 */
	public void setup(Fetcher<Profile> sample, Object[] info, RM...regressions) throws RemoteException {
		List<Object> additionalInfo = Util.newList();
		additionalInfo.add(sample);
		additionalInfo.addAll(Arrays.asList(info));
		
		this.setup((Dataset)null, additionalInfo.toArray(), regressions);
	}

	
	@Override
	public synchronized void unsetup() throws RemoteException {
		super.unsetup();
		this.regressions.clear();
	}


	/*
	 * This method is not marked synchronized because it is called by setup method.
	 */
	@SuppressWarnings("unchecked")
	@Override
	public /*synchronized*/ Object learnStart(Object...info) throws RemoteException {
		List<Object> parameterList = Util.newList();
		boolean success = false;
		for (RM regression : this.regressions) {
			regression.setup((Fetcher<Profile>)sample);
			Object parameter = regression.getParameter();
			parameterList.add(parameter);
			if (parameter != null)
				success = true;
		}
		if (!success)
			return null;
		else
			return parameterList; 
	}

	
	@Override
	public synchronized Object execute(Object input) throws RemoteException {
		List<Object> resultList = Util.newList();
		boolean success = false;
		for (RM regression : this.regressions) {
			Object result = regression.execute(input);
			resultList.add(result);
			if (result != null)
				success = true;
		}
		
		if (!success)
			return null;
		else
			return resultList;
	}

	
	/**
	 * Executing this algorithm by arbitrary input parameter.
	 * @param input arbitrary input parameter.
	 * @return result of execution. Return null if execution is failed.
	 * @throws RemoteException if any error raises.
	 */
	public Object executeIntel(Object...input) throws RemoteException {
		return execute(input);
	}

	
	@Override
	public synchronized Object getParameter() throws RemoteException {
		List<Object> parameterList = Util.newList();
		boolean success = false;
		for (RM regression : this.regressions) {
			Object parameter = regression.getParameter();
			parameterList.add(parameter);
			if (parameter != null)
				success = true;
		}
		
		if (!success)
			return null;
		else
			return parameterList;
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		DataConfig config = super.createDefaultConfig();
		config.put(RM_INDICES_FIELD, RM_INDICES_DEFAULT);
		return config;
	}

	
	@Override
	public synchronized String parameterToShownText(Object parameter, Object... info) throws RemoteException {
		if (parameter == null || !(parameter instanceof List<?>))
			return "";
		
		StringBuffer buffer = new StringBuffer();
		List<?> parameterList = (List<?>)parameter;
		for (int i = 0; i < parameterList.size(); i++) {
			Object element = parameterList.get(i);
			if (i > 0)
				buffer.append(", ");
			String text = this.regressions.get(i).parameterToShownText(element, info);
			buffer.append("{" + text + "}");
		}
		
		return buffer.toString();
	}

	
	@Override
	public synchronized String getDescription() throws RemoteException {
		StringBuffer buffer = new StringBuffer();
		
		for (int i = 0; i < this.regressions.size(); i++) {
			RM regression = this.regressions.get(i);
			if (i > 0)
				buffer.append(", ");
			String text = regression.getDescription();
			buffer.append("{" + text + "}");
		}
		
		return buffer.toString();
	}


	@Override
	public String[] getBaseRemoteInterfaceNames() throws RemoteException {
		return new String[] {MultipleRMRemote.class.getName(), MemoryBasedAlgRemote.class.getName()};
	}

	
	@Override
	public synchronized Object extractResponseValue(Object input) throws RemoteException {
		List<Object> valueList = Util.newList();
		boolean success = false;
		for (RM regression : this.regressions) {
			Object value = regression.extractResponseValue(input);
			valueList.add(value);
			if (value != null)
				success = true;
		}
		
		if (!success)
			return null;
		else
			return valueList;
	}

	
}
