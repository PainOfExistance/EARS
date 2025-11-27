package org.um.feri.ears.examples;

import org.um.feri.ears.algorithms.NumberAlgorithm;
import org.um.feri.ears.algorithms.so.abc.ABC;
import org.um.feri.ears.algorithms.so.avoa.AVOA;
import org.um.feri.ears.algorithms.so.gsa.GSA;
import org.um.feri.ears.algorithms.so.gwo.GWO;
import org.um.feri.ears.algorithms.so.de.jade.JADE;
import org.um.feri.ears.algorithms.so.mvo.MVO;
import org.um.feri.ears.algorithms.so.pso.PSO;
import org.um.feri.ears.algorithms.so.random.RandomSearch;
import org.um.feri.ears.algorithms.so.rsa.RSA;
import org.um.feri.ears.algorithms.so.tlbo.TLBO;
import org.um.feri.ears.algorithms.so.gao.GAO;
import org.um.feri.ears.algorithms.so.woa.WOA;
import org.um.feri.ears.benchmark.Benchmark;
import org.um.feri.ears.benchmark.CEC2015Benchmark;
import org.um.feri.ears.benchmark.CEC2017Benchmark;
import org.um.feri.ears.benchmark.RPUOed30Benchmark;
import org.um.feri.ears.problems.Problem;
import org.um.feri.ears.problems.Solution;
import org.um.feri.ears.problems.StopCriterionException;
import org.um.feri.ears.problems.Task;
import org.um.feri.ears.problems.unconstrained.cec2015.CEC2015;
import org.um.feri.ears.problems.unconstrained.cec2015.F1;

import java.util.ArrayList;

public class SOBenchmarkExample {

    public static void main(String[] args) throws StopCriterionException {
        Benchmark.printInfo = false; //prints one on one results

        //System.out.println("----- CEC2015 Benchmark -----");
        ArrayList<NumberAlgorithm> algorithms = new ArrayList<NumberAlgorithm>();
        //algorithms.add(new ABC());
        //algorithms.add(new GWO());
        //algorithms.add(new TLBO());
        //algorithms.add(new RandomSearch());
        //algorithms.add(new JADE());
        //algorithms.add(new PSO());
        //algorithms.add(new GAO());
//
        //CEC2015Benchmark cec2015Benchmark = new CEC2015Benchmark(); // benchmark with prepared tasks and settings
//
        //cec2015Benchmark.addAlgorithms(algorithms);  // register the algorithms in the benchmark
//
        //cec2015Benchmark.run(10); //start the tournament with 10 runs/repetitions
//
        //System.out.println();
        //System.out.println();
        System.out.println("----- CEC2017 Benchmark -----");

        algorithms = new ArrayList<NumberAlgorithm>();
        algorithms.add(new PSO());
        algorithms.add(new GSA());
        algorithms.add(new TLBO());
        algorithms.add(new MVO());
        algorithms.add(new GWO());
        algorithms.add(new RSA());
        algorithms.add(new AVOA());
        algorithms.add(new GAO());

        CEC2017Benchmark cec2017Benchmark = new CEC2017Benchmark(); // benchmark with prepared tasks and settings

        cec2017Benchmark.addAlgorithms(algorithms);  // register the algorithms in the benchmark

        cec2017Benchmark.run(10); //start the tournament with 10 runs/repetitions

    }
}
