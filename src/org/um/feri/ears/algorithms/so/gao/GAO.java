package org.um.feri.ears.algorithms.so.gao;

import org.um.feri.ears.algorithms.AlgorithmInfo;
import org.um.feri.ears.algorithms.Author;
import org.um.feri.ears.algorithms.NumberAlgorithm;
import org.um.feri.ears.problems.*;
import org.um.feri.ears.util.annotation.AlgorithmParameter;
import org.um.feri.ears.util.random.PredefinedRandom;
import org.um.feri.ears.util.random.RNG;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class GAO extends NumberAlgorithm {

    // Custom random number generator state
    static class RandomState {
        long seed;
        long current;

        RandomState(long seed) {
            this.seed = seed;
            this.current = seed;
        }

        // Custom random number generator (same algorithm as MATLAB)
        public double customRand() {
            long a = 1664525L;
            long c = 1013904223L;
            long m = 1L << 32; // 2^32

            current = (a * current + c) % m;
            return (double) current / (double) m;
        }

        public int customRandInt(int minVal, int maxVal) {
            double r = this.customRand();
            return minVal + (int) Math.floor(r * (maxVal - minVal + 1));
        }
    }

    @AlgorithmParameter(name = "population size")
    private int popSize;
    public boolean isDebug = false;
    private ArrayList<NumberSolution<Double>> population;
    private NumberSolution<Double> bestSolution;
    private RandomState randomState;
    public GAO() {
        this(30);
    }

    public GAO(int popSize) {
        super();
        randomState = new RandomState(42);
        this.popSize = popSize;
        this.isDebug = isDebug;
        au = new Author("matej", "matej.habjanic@student.um.si");
        ai = new AlgorithmInfo("GAO", "Giant Armadillo Optimization",
                "@article{Armadillos,\n" +
                        "  title = {Giant Armadillo Optimization: A New Bio-Inspired Metaheuristic Algorithm for Solving Optimization Problems},\n" +
                        "  volume = {8},\n" +
                        "  ISSN = {2313-7673},\n" +
                        "  url = {http://dx.doi.org/10.3390/biomimetics8080619},\n" +
                        "  DOI = {10.3390/biomimetics8080619},\n" +
                        "  number = {8},\n" +
                        "  journal = {Biomimetics},\n" +
                        "  publisher = {MDPI AG},\n" +
                        "  author = {Alsayyed,  Omar and Hamadneh,  Tareq and Al-Tarawneh,  Hassan and Alqudah,  Mohammad and Gochhait,  Saikat and Leonova,  Irina and Malik,  Om Parkash and Dehghani,  Mohammad},\n" +
                        "  year = {2023},\n" +
                        "  month = dec,\n" +
                        "  pages = {619}\n" +
                        "}"
        );
    }

    @Override
    public NumberSolution<Double> execute(Task<NumberSolution<Double>, DoubleProblem> task) throws StopCriterionException {
        this.task = task;
        initPopulation();

        int maxIterations = 0;
        List<Double> lowerBounds = task.problem.getLowerLimit();
        List<Double> upperBounds = task.problem.getUpperLimit();

        if (task.getStopCriterion() == StopCriterion.ITERATIONS) {
            maxIterations = task.getMaxIterations();
        }

        if (task.getStopCriterion() == StopCriterion.EVALUATIONS) {
            maxIterations = (task.getMaxEvaluations() - popSize) / (popSize * 2);
        }

        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            if(iteration == 100) {
                System.out.println("Iteration: " + iteration + " Best fitness: " + bestSolution.getEval());
                System.exit(0);
            }
            System.out.println("------------");
            System.out.println("Iteration: " + iteration);
            for(int k=0;k<5;k++) {
                System.out.println(Arrays.toString(population.get(k).getVariables().toArray()));
            }
            System.out.println("------------");
            System.out.println();
            System.out.println();

            if (task.isStopCriterion()) {
                break;
            }

            for (int i = 0; i < popSize; i++) {
                if (task.isStopCriterion()) {
                    break;
                }

                NumberSolution<Double> currentAgent = population.get(i);
                double currentFitness = currentAgent.getEval();

                // PHASE 1: Attack on termite mounds (Exploration)
                NumberSolution<Double> termiteMound = selectTermiteMound(currentFitness);

                NumberSolution<Double> newSolution1 = generateNewSolutionPhase1(
                        currentAgent, termiteMound, lowerBounds, upperBounds);

                if (task.isStopCriterion()) break;
                task.eval(newSolution1);

                if (newSolution1.getEval() < currentFitness) {
                    population.set(i, newSolution1);
                    currentFitness = newSolution1.getEval();
                    currentAgent = newSolution1;
                    if (currentFitness < bestSolution.getEval()) {
                        bestSolution = new NumberSolution<>(newSolution1);
                        if (isDebug) {
                            System.out.println("New best solution found: " + Arrays.toString(bestSolution.getVariables().toArray()) +
                                    " with fitness: " + bestSolution.getEval());
                        }
                    }
                }

                // PHASE 2: Digging in termite mounds (Exploitation)
                NumberSolution<Double> newSolution2 = generateNewSolutionPhase2(
                        currentAgent, lowerBounds, upperBounds, iteration);

                if (task.isStopCriterion()) break;
                task.eval(newSolution2);

                if (newSolution2.getEval() <= currentFitness) {
                    population.set(i, newSolution2);

                    if (newSolution2.getEval() < bestSolution.getEval()) {
                        bestSolution = new NumberSolution<>(newSolution2);
                        if (isDebug) {
                            System.out.println("New best solution found: " + Arrays.toString(bestSolution.getVariables().toArray()) +
                                    " with fitness: " + bestSolution.getEval());
                        }
                    }
                }
            }
            iteration++;
        }

        return bestSolution;
    }

    private NumberSolution<Double> selectTermiteMound(double currentFitness) {
        ArrayList<NumberSolution<Double>> betterSolutions = new ArrayList<>();

        for (NumberSolution<Double> solution : population) {
            if (solution.getEval() < currentFitness) {
                betterSolutions.add(solution);
            }
        }

        if (betterSolutions.isEmpty()) {
            return bestSolution;
        }

        int randomIndex = (int) (/*RNG.nextUniform()*/randomState.customRand() * betterSolutions.size());
        return betterSolutions.get(randomIndex);
    }

    private NumberSolution<Double> generateNewSolutionPhase1(NumberSolution<Double> current,
                                                             NumberSolution<Double> termiteMound,
                                                             List<Double> lb, List<Double> ub
    ) {
        List<Double> newPosition = new ArrayList<>();

        for (int j = 0; j < current.getVariables().size(); j++) {
            double I = /*RNG.nextInt(1, 2);//*/randomState.customRandInt(1, 2);
            double r = /*RNG.nextUniform();//*/randomState.customRand();
            double newValue = current.getValue(j) + r * (termiteMound.getValue(j) - I * current.getValue(j));
            newValue = Math.max(newValue, lb.get(j));
            newValue = Math.min(newValue, ub.get(j));
            newPosition.add(newValue);
        }

        return new NumberSolution<Double>(newPosition);
    }

    private NumberSolution<Double> generateNewSolutionPhase2(NumberSolution<Double> current,
                                                             List<Double> lb, List<Double> ub,
                                                             int iteration) {
        List<Double> newPosition = new ArrayList<>();
        double t = iteration + 1;

        for (int j = 0; j < current.getVariables().size(); j++) {
            double newValue = current.getValue(j) + (1 - 2 * /*RNG.nextUniform()*/randomState.customRand()) * (ub.get(j) - lb.get(j)) / t;
            newValue = Math.max(newValue, lb.get(j) / t);
            newValue = Math.min(newValue, ub.get(j) / t);
            newPosition.add(newValue);
        }

        return new NumberSolution<>(newPosition);
    }

    @Override
    public void resetToDefaultsBeforeNewRun() {
        population = null;
        bestSolution = null;
    }

    private void initPopulation() throws StopCriterionException {
        population = new ArrayList<>();
        bestSolution = null;

        List<Double> lowerBounds = task.problem.getLowerLimit();
        List<Double> upperBounds = task.problem.getUpperLimit();
        int dimension = task.problem.getNumberOfDimensions();

        for (int i = 0; i < popSize; i++) {
            if (task.isStopCriterion()) {
                break;
            }

            ArrayList<Double> initialPosition = new ArrayList<>();
            for(int d = 0; d < dimension; d++) {
                double val = this.randomState.customRand() * (upperBounds.get(d) - lowerBounds.get(d)) + lowerBounds.get(d);
                initialPosition.add(val);
            }

            NumberSolution<Double> newSol = new NumberSolution<>();
            newSol.setVariables(initialPosition);
            newSol.setObjectives(task.generateRandomEvaluatedSolution().getObjectives());
            task.eval(newSol);
            population.add(newSol);

            /*NumberSolution<Double> newSolution = task.generateRandomEvaluatedSolution();
            population.add(newSolution);*/

            // Initialize best solution
            if (bestSolution == null || newSol.getEval() < bestSolution.getEval()) {
                bestSolution = new NumberSolution<>(newSol);
                bestSolution.setObjectives(newSol.getObjectives());
            }
        }
    }
}

/*
%%% Designed and Developed by Mohammad Dehghani %%%
%%% Modified with Custom Random Generator (Java-style) %%%

function[Best_score,Best_pos,GAO_curve]=GAO(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fitness)
    % Custom random generator functions
    function randState = initRandomState(seed)
        randState.seed = seed;
        randState.current = seed;
    end

    function [randState, r] = customRand(randState)
        a = 1664525;
        c = 1013904223;
        m = 2^32;

        % Use modulo operation to avoid overflow
        randState.current = mod(a * randState.current + c, m);
        r = double(randState.current) / double(m);
    end

    function [randState, r] = customRandInt(randState, minVal, maxVal)
        [randState, r_val] = customRand(randState);
        r = minVal + floor(r_val * (maxVal - minVal + 1));
    end

    lowerbound = ones(1,dimension).*(lowerbound);    % Lower limit for variables
    upperbound = ones(1,dimension).*(upperbound);    % Upper limit for variables

    % Initialize custom random generator with fixed seed
    randState = initRandomState(42);  % Fixed seed for reproducibility

    %% INITIALIZATION
    % Preallocate X matrix
    X = zeros(SearchAgents, dimension);

    for i=1:SearchAgents
        for j=1:dimension
            % Use custom random generator instead of MATLAB's rand()
            [randState, rand_val] = customRand(randState);
            X(i,j) = lowerbound(j) + rand_val * (upperbound(j) - lowerbound(j));
        end
    end

    % Initialize fitness array
    fit = zeros(1, SearchAgents);

    for i =1:SearchAgents
        L = X(i,:);
        fit(i) = fitness(L);
    end

    %% MAIN OPTIMIZATION LOOP
    % Preallocate convergence curve
    best_so_far = zeros(1, Max_iterations);
    average = zeros(1, Max_iterations);

    for t=1:Max_iterations
        fprintf('=== Iteration %d ===\n', t);

        %% Update: BEST proposed solution
        [Fbest, blocation] = min(fit);

        if t==1
            xbest = X(blocation,:);      % Optimal location
            fbest = Fbest;                % The optimization objective function
        elseif Fbest < fbest
            fbest = Fbest;
            xbest = X(blocation,:);
        end


        %% Process each search agent
        for i=1:5
            fprintf('    Position: ');
            fprintf('%6.4f ', X(i,:));
            fprintf('\n');

            %% Phase 1: Attack on termite mounds (exploration phase)
            TM_location = find(fit < fit(i));  % based on Eq(4)
            if isempty(TM_location)
                STM = xbest;
            else
                % Use customRandInt instead of randperm
                [randState, K_idx] = customRandInt(randState, 1, length(TM_location));
                K = TM_location(K_idx);
                STM = X(K,:);
            end

            % Use custom random generator
            [randState, rand_val1] = customRand(randState);
            I = round(1 + rand_val1);

            [randState, rand_val2] = customRand(randState);
            X_new_P1 = X(i,:) + rand_val2 * (STM - I * X(i,:));  % Eq(5)

            % Boundary control
            for d=1:dimension
                if X_new_P1(d) < lowerbound(d)
                    X_new_P1(d) = lowerbound(d);
                elseif X_new_P1(d) > upperbound(d)
                    X_new_P1(d) = upperbound(d);
                end
            end

            % Update position based on Eq (6)
            L = X_new_P1;
            fit_new_P1 = fitness(L);

            if fit_new_P1 < fit(i)
                X(i,:) = X_new_P1;
                fit(i) = fit_new_P1;
            end

            %% Phase 2: Digging in termite mounds (exploitation phase)
            % Use custom random generator
            [randState, r_val] = customRand(randState);
            X_new_P2 = X(i,:) + (1 - 2 * r_val) * (upperbound - lowerbound) / t;  % Eq(7)

            % Boundary control for Phase 2
            for d=1:dimension
                lowerbound_t = lowerbound(d)/t;
                upperbound_t = upperbound(d)/t;

                if X_new_P2(d) < lowerbound_t
                    X_new_P2(d) = lowerbound_t;
                elseif X_new_P2(d) > upperbound_t
                    X_new_P2(d) = upperbound_t;
                end
            end

            % Updating X_i using (8)
            L = X_new_P2;
            f_new = fitness(L);

            if f_new <= fit(i)
                X(i,:) = X_new_P2;
                fit(i) = f_new;

                if f_new < fbest
                    xbest = X_new_P2;
                    fbest = f_new;
                end
            else
            end
        end  % End agent loop

        best_so_far(t) = fbest;
        average(t) = mean(fit);

        fprintf('\n');

        %% Special handling for 100th iteration
        if t == 20
            fprintf('\n=========================================\n');
            fprintf('!!! ITERATION 20 - SHOWING ALL AGENTS !!!\n');
            fprintf('=========================================\n\n');

            fprintf('All Agents at Iteration 20:\n');
            for i=1:SearchAgents
                fprintf('Agent %2d: Fitness = %6.4f, Position = ', i, fit(i));
                fprintf('%6.4f ', X(i,:));
                fprintf('\n');
            end

            fprintf('\nFitness Statistics:\n');
            fprintf('Best Fitness: %6.4f\n', fbest);
            fprintf('Worst Fitness: %6.4f\n', max(fit));
            fprintf('Average Fitness: %6.4f\n', mean(fit));
            fprintf('Std Dev: %6.4f\n', std(fit));

            fprintf('\nBreaking execution...\n');
            break;  % Break the loop at iteration 100
        end

    end  % End main loop

    % Truncate the convergence curve if we broke early
    if t < Max_iterations
        best_so_far = best_so_far(1:t);
        average = average(1:t);
    end

    % Assign final outputs
    Best_score = fbest;
    Best_pos = xbest;
    GAO_curve = best_so_far;
end

% Example call with sphere function
fitness = @(x) sum(x.^2);
[Best_score, Best_pos, GAO_curve] = GAO(30, 100, -100, 100, 5, fitness);
 */