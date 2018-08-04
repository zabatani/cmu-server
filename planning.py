import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy

N = 5
n = 2 ** N
gamma = 1
mu = np.array([0.6, 0.5, 0.3, 0.7, 0.1])
c = np.array([1, 4, 6, 2, 9])
figure_counter = 0


def get_value_func(pi):
	P_pi = np.zeros([n, n])
	c_pi = np.zeros([n])
	for current_state in range(n):
		action = pi[current_state]
		if action >= 0:
			next_state = int2bin(current_state)
			next_state[action] = '1'
			next_state = bin2int(next_state)
			P_pi[current_state, next_state] = mu[action]
			P_pi[current_state, current_state] = 1 - mu[action]
			c_pi[current_state] = np.sum(c[int2bin(current_state) == '0'])
		else:  # Terminal state
			P_pi[current_state, current_state] = 0
			c_pi[current_state] = 0

	value_func = np.linalg.inv(np.eye(n) - gamma * P_pi) @ c_pi
	return value_func


def max_cost_policy(state):
	if state == n - 1:
		return -1
	available_jobs = int2bin(state) == '0'
	next_job = np.argmax(c[available_jobs])
	next_job = np.arange(N)[available_jobs][next_job]

	return next_job


def muc_rule_policy(state):
	if state == n - 1:
		return -1
	available_jobs = int2bin(state) == '0'
	next_job = np.argmax(c[available_jobs] * mu[available_jobs])
	next_job = np.arange(N)[available_jobs][next_job]

	return next_job


def policy_iteration(policy):
	policy = deepcopy(policy)
	prev_policy_value = None
	initial_state_values = []
	while True:
		policy_value = get_value_func(policy)
		initial_state_values.append(policy_value[0])
		if np.array_equal(policy_value, prev_policy_value):
			break

		for state in range(n):
			min_value = np.inf
			for i in range(N):
				next_state_bin = int2bin(state)
				if next_state_bin[i] == '0':
					next_state_bin[i] = '1'
					next_state = bin2int(next_state_bin)
					value = mu[i] * policy_value[next_state] + (1 - mu[i]) * policy_value[state]
					if value < min_value:
						min_value = value
						policy[state] = i

		prev_policy_value = policy_value

	plot(initial_state_values)
	return policy


def simulator(state, action):
	cost = np.sum(c[int2bin(state) == '0'])  # Independent of the action
	if np.random.random() <= mu[action]:
		next_state = int2bin(state)
		next_state[action] = '1'
		next_state = bin2int(next_state)
	else:
		next_state = state

	return cost, next_state


def Q_learning(optimal_value_func, alpha_type=2, epsilon=0.1, epochs=10000):
	epochs = epochs
	visits = defaultdict(int)
	q_func = np.zeros([n, N])

	initial_diff = []
	max_diff = []

	for epoch in range(epochs):
		#state = np.random.randint(n)  # No need thanks to the usage of epsilon-greedy
		state = 0
		while True:
			if state == n - 1:
				break
			action = get_action(state, q_func[state, :], epsilon)  # State is only being transfer in-order to avoid invalid actions
			cost, next_state = simulator(state, action)
			visits[state] += 1

			if next_state == n - 1:
				d = cost - q_func[state, action]
			else:
				d = cost + gamma * np.min(q_func[next_state, int2bin(next_state) == '0']) - q_func[state, action]

			if alpha_type == 0:
				alpha = 1 / visits[state]
			elif alpha_type == 1:
				alpha = 0.01
			else:
				alpha = 10 / (100 + visits[state])

			q_func[state, action] += alpha * d

			state = next_state
		if epoch % 100 == 0:
			policy = []
			for i in range(n - 1):
				available_jobs = int2bin(i) == '0'
				# q_func = q_func[i, :]
				action = np.argmin(q_func[i, available_jobs])
				action = np.arange(N)[available_jobs][action]
				policy.append(action)
			policy.append(-1)
			value_func = get_value_func(policy)
			initial_diff.append(np.abs(optimal_value_func[0] - q_func[0,policy[0]]))
			max_diff.append(np.max(np.abs(optimal_value_func - value_func)))

	plot(initial_diff, title="Initial state difference - Q-Learning alpha_type={}, epsilon={}".format(alpha_type, epsilon))
	plot(max_diff, title="Max state difference - Q-Learning alpha_type={}, epsilon={}".format(alpha_type, epsilon))

	return q_func


def get_action(state, q_func, epsilon):
	# available_jobs = int2bin(state) == '0'
	# next_job = np.argmax(c[available_jobs])
	# next_job = np.arange(N)[available_jobs][next_job]
	available_jobs = int2bin(state) == '0'
	if np.random.random() > epsilon:
		action = np.argmin(q_func[available_jobs])
	else:
		action = np.random.randint(sum(available_jobs))

	action = np.arange(N)[available_jobs][action]
	return action


def TD_lambda(policy, true_value_func, alpha_type=2, lamb=0, epochs=20000):
	epochs = epochs
	visits = defaultdict(int)
	value_func = np.zeros(n)

	initial_diff = np.zeros(epochs)
	max_diff = np.zeros(epochs)

	for epoch in range(epochs):
		state = np.random.randint(n)
		state_history = []
		while True:
			if state == n - 1:
				break
			action = policy[state]
			cost, next_state = simulator(state, action)
			visits[state] += 1
			state_history.append(state)

			d = cost + gamma * value_func[next_state] - value_func[state]

			m = len(state_history)
			for i in range(m):
				if alpha_type == 0:
					alpha = 1 / visits[state_history[i]]
				elif alpha_type == 1:
					alpha = 0.01
				else:
					alpha = 10 / (100 + visits[state_history[i]])
				value_func[state_history[i]] += alpha * (lamb ** (m - 1 - i)) * d

			state = next_state

		initial_diff[epoch] = np.abs(true_value_func[0] - value_func[0])
		max_diff[epoch] = np.max(np.abs(true_value_func - value_func))

	plot(initial_diff, title="Initial state difference - TD({})".format(lamb))
	plot(max_diff, title="Max state difference - TD({})".format(lamb))
	return value_func


def int2bin(x):
	return np.array(list(np.binary_repr(x, N)))


def bin2int(x):
	return int(''.join(x.tolist()), 2)


def plot(y, type="line", title=None):
	plt.figure(figsize=(15, 4))
	x = list(range(len(y)))
	if type == "line":
		plt.plot(x, y)
	if type == "bar":
		plt.bar(x, y)
	if title:
		plt.title(title)
	global figure_counter
	plt.savefig('plots/figure-{}.png'.format(figure_counter))
	figure_counter += 1


def main():
	max_cost_pi = [max_cost_policy(i) for i in range(n)]
	plot(max_cost_pi, type="bar", title="Maximum Cost Policy")

	max_cost_iterated_pi = policy_iteration(max_cost_pi)

	muc_rule_pi = [muc_rule_policy(i) for i in range(n)]

	max_cost_value_func = get_value_func(max_cost_pi)
	max_cost_iterated_value_func = get_value_func(max_cost_iterated_pi)
	muc_rule_value_func = get_value_func(muc_rule_pi)

	plot(max_cost_value_func, type="bar", title="Max Cost Value Function")
	plot(max_cost_iterated_value_func, type="bar", title="Max Cost Iterated Value Function")
	plot(muc_rule_value_func, type="bar", title="Mu-C Rule Value Function")

	TD_lambda(max_cost_pi, max_cost_value_func, lamb=0, alpha_type=0)
	TD_lambda(max_cost_pi, max_cost_value_func, lamb=0.2)
	TD_lambda(max_cost_pi, max_cost_value_func, lamb=0.6)
	TD_lambda(max_cost_pi, max_cost_value_func, lamb=0.8)

	Q_learning(muc_rule_value_func, epsilon=0.01, epochs=100000, alpha_type=2)

	plt.show()


if __name__ == "__main__":
	main()
