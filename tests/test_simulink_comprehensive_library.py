"""Comprehensive test suite verifying every single block in the Terminus Simulink Library."""

import unittest
import math
import numpy as np

from Engines.Dynamic_System.blocks import *
from Engines.Dynamic_System.scheduler import SystemDiagram
from Engines.Dynamic_System.ode_solver import DynamicSystemSimulator


class TestSimulinkLibraryComprehensive(unittest.TestCase):

    # =========================================================================
    # 1. CONTINUOUS BLOCKS
    # =========================================================================

    def test_integrator_and_limited(self):
        # Continuous integrator
        integ = IntegratorBlock("Int1", initial_condition=1.0)
        self.assertEqual(integ.compute_output(0.0)[0], 1.0)
        integ.inputs = [5.0]
        deriv = integ.compute_derivatives(0.0)
        self.assertEqual(deriv[0], 5.0)

        # Limited integrator with anti-windup clamping
        integ_lim = IntegratorLimitedBlock("IntLim", initial_condition=10.0, lower_limit=-5.0, upper_limit=10.0)
        integ_lim.inputs = [2.0]
        # Pushing above upper limit => derivative should clamp to 0.0
        self.assertEqual(integ_lim.compute_derivatives(0.0)[0], 0.0)
        # Pulling down => derivative should be negative
        integ_lim.inputs = [-2.0]
        self.assertEqual(integ_lim.compute_derivatives(0.0)[0], -2.0)

    def test_second_order_integrator_and_limited(self):
        soi = SecondOrderIntegratorBlock("SOI", ic_x=0.0, ic_dx=2.0)
        out = soi.compute_output(0.0)
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[1], 2.0)
        soi.inputs = [9.81]
        derivs = soi.compute_derivatives(0.0)
        self.assertEqual(derivs[0], 2.0)  # dx/dt = v
        self.assertEqual(derivs[1], 9.81) # dv/dt = a

        soi_lim = SecondOrderIntegratorLimitedBlock("SOILim", ic_x=10.0, ic_dx=5.0, upper_limit_x=10.0, upper_limit_dx=5.0)
        soi_lim.inputs = [10.0]
        d_lim = soi_lim.compute_derivatives(0.0)
        self.assertEqual(d_lim[0], 0.0)
        self.assertEqual(d_lim[1], 0.0)

    def test_derivative_block(self):
        deriv = DerivativeBlock("Deriv", tau=0.05)
        deriv.inputs = [10.0]
        self.assertEqual(deriv.compute_output(0.0)[0], 200.0)  # (10 - 0) / 0.05
        d = deriv.compute_derivatives(0.0)
        self.assertEqual(d[0], 200.0)

    def test_transfer_function_and_zero_pole(self):
        # H(s) = 2 / (s + 2)
        tf = TransferFunctionBlock("TF", num=[2.0], den=[1.0, 2.0])
        self.assertEqual(tf.num_states, 1)
        tf.inputs = [1.0]
        y0 = tf.compute_output(0.0)[0]
        self.assertAlmostEqual(y0, 0.0)

        # Zero-Pole: H(s) = 2 * (s + 1) / (s + 2)
        zp = ZeroPoleBlock("ZP", zeros=[-1.0], poles=[-2.0], gain=2.0)
        self.assertEqual(zp.num_states, 1)
        zp.inputs = [3.0]
        self.assertIsNotNone(zp.compute_output(0.0)[0])

    def test_descriptor_state_space(self):
        E = np.array([[2.0, 0.0], [0.0, 1.0]])
        A = np.array([[-2.0, 1.0], [0.0, -1.0]])
        B = np.array([[1.0], [2.0]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])
        dss = DescriptorStateSpaceBlock("DSS", E=E, A=A, B=B, C=C, D=D)
        dss.inputs = [1.0]
        derivs = dss.compute_derivatives(0.0)
        self.assertEqual(len(derivs), 2)
        self.assertAlmostEqual(derivs[0], 0.5)  # (1.0) / 2.0
        self.assertAlmostEqual(derivs[1], 2.0)

    def test_delays_continuous(self):
        td = TransportDelayBlock("TD", delay_time=0.5, initial_output=3.0)
        td.inputs = [10.0]
        self.assertEqual(td.compute_output(0.0)[0], 3.0)

        vtd = VariableTimeDelayBlock("VTD", max_delay=5.0, initial_output=0.0)
        vtd.inputs = [10.0, 1.0]
        self.assertEqual(vtd.compute_output(0.0)[0], 0.0)

    def test_first_order_hold(self):
        foh = FirstOrderHoldBlock("FOH", sample_time=0.1)
        foh.inputs = [10.0]
        self.assertEqual(foh.compute_output(0.0)[0], 10.0)

    def test_pid_2dof(self):
        pid2 = PIDController2DOFBlock("PID2DOF", kp=2.0, ki=1.0, kd=0.0, b=0.8, c=0.0)
        pid2.inputs = [10.0, 2.0]  # r=10, y=2 -> b*r - y = 0.8*10 - 2 = 6.0 -> P term = 12.0
        out = pid2.compute_output(0.0)[0]
        self.assertAlmostEqual(out, 12.0)

        # With derivative action
        pid2_d = PIDController2DOFBlock("PID2DOF_D", kp=2.0, ki=0.0, kd=0.5, n_filter=10.0, b=1.0, c=1.0)
        pid2_d.inputs = [10.0, 0.0]  # c*r - y = 10, d_term = 0.5 * 10 * 10 = 50, p_term = 20
        out_d = pid2_d.compute_output(0.0)[0]
        self.assertAlmostEqual(out_d, 70.0)

    # =========================================================================
    # 2. DISCRETE BLOCKS
    # =========================================================================

    def test_discrete_delays_and_difference(self):
        # Unit Delay
        ud = UnitDelayBlock("UD", sample_time=0.1, initial_condition=5.0)
        ud.inputs = [10.0]
        self.assertEqual(ud.compute_output(0.0)[0], 5.0)
        ud.compute_output(0.15)
        self.assertEqual(ud.compute_output(0.25)[0], 10.0)

        # Difference
        diff = DifferenceBlock("Diff", sample_time=0.1, initial_condition=0.0)
        diff.inputs = [10.0]
        self.assertEqual(diff.compute_output(0.0)[0], 10.0)
        diff.inputs = [12.0]
        self.assertEqual(diff.compute_output(0.15)[0], 2.0)

        # Discrete Derivative
        dderiv = DiscreteDerivativeBlock("DDeriv", sample_time=0.1, initial_condition=0.0)
        dderiv.inputs = [10.0]
        self.assertEqual(dderiv.compute_output(0.0)[0], 100.0)

        # Tapped Delay
        td = TappedDelayBlock("TD", num_taps=3, sample_time=0.1, initial_condition=0.0)
        td.inputs = [1.0]
        out_t = td.compute_output(0.0)
        self.assertEqual(len(out_t), 3)

        # Resettable Delay & Variable Integer Delay
        rd = ResettableDelayBlock("RD", delay_length=2, sample_time=0.1)
        rd.inputs = [10.0, 0.0]
        rd.compute_output(0.0)
        self.assertIsNotNone(rd.compute_output(0.1)[0])

        vid = VariableIntegerDelayBlock("VID", max_delay=10, sample_time=0.1)
        vid.inputs = [5.0, 2]
        self.assertIsNotNone(vid.compute_output(0.0)[0])

    def test_discrete_filters_and_tf(self):
        # FIR Filter
        fir = DiscreteFIRFilterBlock("FIR", coefficients=[0.5, 0.5], sample_time=0.1)
        fir.inputs = [10.0]
        self.assertAlmostEqual(fir.compute_output(0.0)[0], 5.0)

        # IIR Filter
        iir = DiscreteFilterBlock("IIR", num=[1.0], den=[1.0, -0.5], sample_time=0.1)
        iir.inputs = [2.0]
        self.assertAlmostEqual(iir.compute_output(0.0)[0], 2.0)

        # Lead/Lag and Real Zero
        lead_lag = TransferFcnLeadOrLagBlock("LL", a=0.5, b=0.8, sample_time=0.1)
        lead_lag.inputs = [1.0]
        self.assertIsNotNone(lead_lag.compute_output(0.0)[0])

        real_zero = TransferFcnRealZeroBlock("RZ", zero=0.5, gain=2.0, sample_time=0.1)
        real_zero.inputs = [1.0]
        self.assertIsNotNone(real_zero.compute_output(0.0)[0])

        # First Order Discrete TF
        tf1 = TransferFcnFirstOrderBlock("TF1", gain=2.0, pole=0.5, sample_time=0.1)
        tf1.inputs = [1.0]
        self.assertIsNotNone(tf1.compute_output(0.0)[0])

        # Direct Form II
        df2 = TransferFcnDirectFormIIBlock("DF2", num=[1.0, 0.5], den=[1.0, -0.5], sample_time=0.1)
        df2.inputs = [2.0]
        self.assertIsNotNone(df2.compute_output(0.0)[0])

        # Discrete Time Integrator
        dint = DiscreteTimeIntegratorBlock("DInt", sample_time=0.1, integration_method="Forward Euler")
        dint.inputs = [5.0]
        self.assertEqual(dint.compute_output(0.0)[0], 0.0)
        dint.compute_output(0.15)
        self.assertAlmostEqual(dint.compute_output(0.25)[0], 1.0)

        # Discrete PID 2DOF
        dpid2 = DiscretePIDController2DOFBlock("DPID2", kp=1.0, ki=1.0, kd=0.0, b=1.0, sample_time=0.1)
        dpid2.inputs = [5.0, 2.0]
        self.assertAlmostEqual(dpid2.compute_output(0.0)[0], 3.0)

    # =========================================================================
    # 3. DISCONTINUITIES
    # =========================================================================

    def test_discontinuities(self):
        # Dynamic Saturation
        sat_dyn = SaturationDynamicBlock("SatDyn")
        sat_dyn.inputs = [15.0, 10.0, -10.0]
        self.assertEqual(sat_dyn.compute_output(0.0)[0], 10.0)

        # Dynamic Rate Limiter
        rl_dyn = RateLimiterDynamicBlock("RLDyn", initial_output=0.0)
        rl_dyn.inputs = [100.0, 10.0, -10.0]
        rl_dyn.compute_output(0.0)
        out = rl_dyn.compute_output(0.1)[0]
        self.assertAlmostEqual(out, 1.0)  # 0 + 10 * 0.1

        # Dynamic Dead Zone
        dz_dyn = DeadZoneDynamicBlock("DZDyn")
        dz_dyn.inputs = [5.0, 2.0, -2.0]
        self.assertEqual(dz_dyn.compute_output(0.0)[0], 3.0)

        # Hit Crossing
        hc = HitCrossingBlock("HC", hit_crossing_offset=0.0, direction="rising")
        hc.inputs = [-1.0]
        self.assertEqual(hc.compute_output(0.0)[0], 0.0)
        hc.inputs = [1.0]
        self.assertEqual(hc.compute_output(0.1)[0], 1.0)

        # PWM
        pwm = PWMBlock("PWM", frequency=10.0, amplitude=5.0)
        pwm.inputs = [0.5]  # 50% duty
        self.assertEqual(pwm.compute_output(0.01)[0], 5.0)
        self.assertEqual(pwm.compute_output(0.08)[0], 0.0)

        # Variable Pulse Gen
        vpg = VariablePulseGeneratorBlock("VPG")
        vpg.inputs = [1.0, 0.5, 5.0]
        self.assertEqual(vpg.compute_output(0.1)[0], 5.0)

        # Wrap To Zero
        wtz = WrapToZeroBlock("WTZ", threshold=10.0)
        wtz.inputs = [12.0]
        self.assertEqual(wtz.compute_output(0.0)[0], 0.0)
        wtz.inputs = [8.0]
        self.assertEqual(wtz.compute_output(0.0)[0], 8.0)

    # =========================================================================
    # 4. LOGIC AND BIT OPERATIONS
    # =========================================================================

    def test_logic_and_bit_ops(self):
        # Logical Operator
        log_and = LogicalOperatorBlock("LogAND", operator="AND")
        log_and.inputs = [1.0, 0.0]
        self.assertEqual(log_and.compute_output(0.0)[0], 0.0)

        log_xor = LogicalOperatorBlock("LogXOR", operator="XOR")
        log_xor.inputs = [1.0, 0.0]
        self.assertEqual(log_xor.compute_output(0.0)[0], 1.0)

        # Relational Operator
        rel = RelationalOperatorBlock("Rel", operator="<=")
        rel.inputs = [3.0, 5.0]
        self.assertEqual(rel.compute_output(0.0)[0], 1.0)

        # Compare To Constant / Zero
        c2c = CompareToConstantBlock("C2C", operator=">", constant=5.0)
        c2c.inputs = [7.0]
        self.assertEqual(c2c.compute_output(0.0)[0], 1.0)

        # Interval Test & Dynamic Interval Test
        it = IntervalTestBlock("IT", lower_bound=0.0, upper_bound=10.0, interval_type="Closed")
        it.inputs = [5.0]
        self.assertEqual(it.compute_output(0.0)[0], 1.0)
        it.inputs = [15.0]
        self.assertEqual(it.compute_output(0.0)[0], 0.0)

        it_dyn = IntervalTestDynamicBlock("ITDyn")
        it_dyn.inputs = [5.0, 10.0, 0.0]
        self.assertEqual(it_dyn.compute_output(0.0)[0], 1.0)

        # Bit Operations
        bset = BitSetBlock("BSet", bit_position=2)
        bset.inputs = [1.0]  # 1 | (1<<2) = 5
        self.assertEqual(bset.compute_output(0.0)[0], 5.0)

        bclr = BitClearBlock("BClr", bit_position=0)
        bclr.inputs = [5.0]  # 5 & ~1 = 4
        self.assertEqual(bclr.compute_output(0.0)[0], 4.0)

        # Extract bits & Shift
        ext_b = ExtractBitsBlock("ExtB", start_bit=1, end_bit=3)
        ext_b.inputs = [14.0]  # binary 1110 -> bits 1..3 = 111 = 7
        self.assertEqual(ext_b.compute_output(0.0)[0], 7.0)

        shift = ShiftArithmeticBlock("Shift", shift_bits=2, direction="Left")
        shift.inputs = [3.0]  # 3 << 2 = 12
        self.assertEqual(shift.compute_output(0.0)[0], 12.0)

        # Bit to Int / Int to Bit
        b2i = BitToIntegerConverterBlock("B2I", num_bits=3, lsb_first=False)
        b2i.inputs = [1, 0, 1]  # 5 in binary
        self.assertEqual(b2i.compute_output(0.0)[0], 5.0)

        i2b = IntegerToBitConverterBlock("I2B", num_bits=3, lsb_first=False)
        i2b.inputs = [5.0]
        self.assertEqual(i2b.compute_output(0.0), [1.0, 0.0, 1.0])

        # Combinatorial Logic Truth Table
        tt = CombinatorialLogicBlock("TT", truth_table=[[0.0], [0.0], [0.0], [1.0]])
        tt.inputs = [1.0, 1.0]  # row 3
        self.assertEqual(tt.compute_output(0.0)[0], 1.0)

        # Edge Detectors
        det_rise = DetectRisePositiveBlock("DetRise")
        det_rise.inputs = [-1.0]
        det_rise.compute_output(0.0)
        det_rise.inputs = [1.0]
        self.assertEqual(det_rise.compute_output(0.1)[0], 1.0)

        det_fall = DetectFallNegativeBlock("DetFall")
        det_fall.inputs = [1.0]
        det_fall.compute_output(0.0)
        det_fall.inputs = [-1.0]
        self.assertEqual(det_fall.compute_output(0.1)[0], 1.0)

    # =========================================================================
    # 5. MATH OPERATIONS
    # =========================================================================

    def test_math_operations(self):
        # Divide
        div = DivideBlock("Div")
        div.inputs = [10.0, 2.0]
        self.assertEqual(div.compute_output(0.0)[0], 5.0)

        # Dot product
        dot = DotProductBlock("Dot")
        dot.inputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(dot.compute_output(0.0)[0], 32.0)

        # Math function (pow)
        mf_pow = MathFunctionBlock("MFPow", function="pow")
        mf_pow.inputs = [2.0, 3.0]
        self.assertEqual(mf_pow.compute_output(0.0)[0], 8.0)

        # Rounding function
        rf = RoundingFunctionBlock("RF", function="floor")
        rf.inputs = [3.7]
        self.assertEqual(rf.compute_output(0.0)[0], 3.0)

        # Trigonometric
        trig = TrigonometricFunctionBlock("Trig", function="cos")
        trig.inputs = [0.0]
        self.assertEqual(trig.compute_output(0.0)[0], 1.0)

        # Polynomial: P(u) = 2*u^2 + 3*u + 1
        poly = PolynomialBlock("Poly", coefficients=[2.0, 3.0, 1.0])
        poly.inputs = [2.0]  # 2*4 + 3*2 + 1 = 15
        self.assertEqual(poly.compute_output(0.0)[0], 15.0)

        # Complex conversions
        c2ma = ComplexToMagnitudeAngleBlock("C2MA", output_in_degrees=True)
        c2ma.inputs = [complex(0.0, 1.0)]
        out_c = c2ma.compute_output(0.0)
        self.assertAlmostEqual(out_c[0], 1.0)  # magnitude
        self.assertAlmostEqual(out_c[1], 90.0)  # phase

        # Increment / Decrement
        inc_rw = IncrementRealWorldBlock("IncRW")
        inc_rw.inputs = [5.0]
        self.assertEqual(inc_rw.compute_output(0.0)[0], 6.0)

        dec_z = DecrementToZeroBlock("DecZ")
        dec_z.inputs = [0.5]
        self.assertEqual(dec_z.compute_output(0.0)[0], 0.0)

        # MinMax Running Resettable
        mm_run = MinMaxRunningResettableBlock("MMRun", function="max")
        mm_run.inputs = [10.0, 0.0]
        mm_run.compute_output(0.0)
        mm_run.inputs = [5.0, 0.0]
        self.assertEqual(mm_run.compute_output(0.1)[0], 10.0)

    # =========================================================================
    # 6. MATRIX OPERATIONS
    # =========================================================================

    def test_matrix_operations(self):
        # Diagonal create and extract
        cdm = CreateDiagonalMatrixBlock("CDM")
        cdm.inputs = [[1.0, 2.0, 3.0]]
        diag_m = cdm.compute_output(0.0)[0]
        self.assertTrue(np.array_equal(diag_m, np.diag([1.0, 2.0, 3.0])))

        edm = ExtractDiagonalBlock("EDM")
        edm.inputs = [diag_m]
        self.assertTrue(np.array_equal(edm.compute_output(0.0)[0], [1.0, 2.0, 3.0]))

        # Cross Product
        cp = CrossProductBlock("CP")
        cp.inputs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        self.assertTrue(np.array_equal(cp.compute_output(0.0)[0], [0.0, 0.0, 1.0]))

        # IsSymmetric & IsTriangular & IsHermitian
        is_sym = IsSymmetricBlock("IsSym")
        is_sym.inputs = [np.array([[1.0, 2.0], [2.0, 1.0]])]
        self.assertEqual(is_sym.compute_output(0.0)[0], 1.0)

        is_tri = IsTriangularBlock("IsTri", triangular_type="Upper")
        is_tri.inputs = [np.array([[1.0, 2.0], [0.0, 1.0]])]
        self.assertEqual(is_tri.compute_output(0.0)[0], 1.0)

        is_herm = IsHermitianBlock("IsHerm")
        is_herm.inputs = [np.array([[1.0, 1.0j], [-1.0j, 2.0]])]
        self.assertEqual(is_herm.compute_output(0.0)[0], 1.0)

        # Matrix Multiply
        pmm = ProductMatrixMultiplyBlock("PMM")
        pmm.inputs = [np.array([[1.0, 2.0]]), np.array([[3.0], [4.0]])]
        self.assertEqual(pmm.compute_output(0.0)[0][0, 0], 11.0)

        # Submatrix and Reshape
        sub_m = SubmatrixBlock("SubM", row_span=(0, 0), col_span=(0, 1))
        sub_m.inputs = [np.array([[10, 20], [30, 40]])]
        self.assertEqual(sub_m.compute_output(0.0)[0].shape, (1, 2))

        resh = ReshapeBlock("Resh", output_shape=(4,))
        resh.inputs = [np.array([[1, 2], [3, 4]])]
        self.assertEqual(len(resh.compute_output(0.0)[0]), 4)

    # =========================================================================
    # 7. LOOKUP TABLES
    # =========================================================================

    def test_lookup_tables(self):
        # 2D Lookup Table
        x = [0.0, 1.0]
        y = [0.0, 1.0]
        table = [[0.0, 10.0], [20.0, 30.0]]
        lut2d = LookupTable2DBlock("LUT2D", x_data=x, y_data=y, table_data=table)
        lut2d.inputs = [0.5, 0.5]
        self.assertAlmostEqual(lut2d.compute_output(0.0)[0], 15.0)

        # Prelookup & Interpolation using Prelookup
        pre = PrelookupBlock("Pre", breakpoints=[0.0, 10.0, 20.0])
        pre.inputs = [5.0]
        out_pre = pre.compute_output(0.0)
        self.assertEqual(out_pre[0], 0.0)  # index 0
        self.assertAlmostEqual(out_pre[1], 0.5)  # fraction 0.5

        interp_blk = InterpolationUsingPrelookupBlock("Interp", table_data=[0.0, 100.0, 200.0])
        interp_blk.inputs = [out_pre[0], out_pre[1]]
        self.assertAlmostEqual(interp_blk.compute_output(0.0)[0], 50.0)

        # Sine Lookup
        sine_lut = SineLookupBlock("SineLUT", num_points=50)
        sine_lut.inputs = [math.pi / 2.0]
        self.assertAlmostEqual(sine_lut.compute_output(0.0)[0], 1.0, delta=0.01)

    # =========================================================================
    # 8. SOURCES & SINKS
    # =========================================================================

    def test_sources_and_sinks(self):
        # Clock & Digital Clock
        clk = ClockBlock("Clk")
        self.assertEqual(clk.compute_output(2.5)[0], 2.5)

        dclk = DigitalClockBlock("DClk", sample_time=0.5)
        self.assertEqual(dclk.compute_output(2.3)[0], 2.0)

        # Counter Free Running
        cfr = CounterFreeRunningBlock("CFR", num_bits=2, sample_time=1.0)
        self.assertEqual(cfr.compute_output(5.0)[0], 1.0)  # 5 % 4 = 1

        # Chirp
        chirp = ChirpSignalBlock("Chirp", f0=1.0, f1=10.0, t1=5.0)
        self.assertIsNotNone(chirp.compute_output(0.5)[0])

        # Sinks: Display, Terminator, ToWorkspace
        disp = DisplaySinkBlock("Disp")
        disp.inputs = [42.0]
        self.assertEqual(disp.compute_output(0.0)[0], 42.0)

        towork = ToWorkspaceBlock("ToWork", variable_name="var1")
        towork.inputs = [99.0]
        towork.compute_output(0.0)
        self.assertEqual(len(towork.get_data()["time"]), 1)

    # =========================================================================
    # 9. SIGNAL ROUTING & BUSES
    # =========================================================================

    def test_routing_and_buses(self):
        # Mux & Demux
        mux = MuxBlock("Mux", num_inputs=2)
        mux.inputs = [1.0, 2.0]
        mux_out = mux.compute_output(0.0)[0]
        self.assertTrue(np.array_equal(mux_out, [1.0, 2.0]))

        demux = DemuxBlock("Demux", num_outputs=2)
        demux.inputs = [mux_out]
        demux_out = demux.compute_output(0.0)
        self.assertEqual(demux_out[0], 1.0)
        self.assertEqual(demux_out[1], 2.0)

        # Bus Creator and Selector
        bc = BusCreatorBlock("BC", signal_names=["voltage", "current"])
        bc.inputs = [12.0, 2.5]
        bus_out = bc.compute_output(0.0)[0]
        self.assertEqual(bus_out, {"voltage": 12.0, "current": 2.5})

        bs = BusSelectorBlock("BS", output_signals=["current", "voltage"])
        bs.inputs = [bus_out]
        bs_out = bs.compute_output(0.0)
        self.assertEqual(bs_out[0], 2.5)
        self.assertEqual(bs_out[1], 12.0)

        # Goto & From
        goto = GotoBlock("GotoA", tag="MOTOR_SPEED")
        from_b = FromBlock("FromA", tag="MOTOR_SPEED")
        goto.inputs = [3000.0]
        goto.compute_output(0.0)
        self.assertEqual(from_b.compute_output(0.0)[0], 3000.0)

        # Data Store Memory / Read / Write
        dsm = DataStoreMemoryBlock("DSM", data_store_name="VAR_X", initial_value=10.0)
        dsr = DataStoreReadBlock("DSR", data_store_name="VAR_X")
        dsw = DataStoreWriteBlock("DSW", data_store_name="VAR_X")
        self.assertEqual(dsr.compute_output(0.0)[0], 10.0)
        dsw.inputs = [50.0]
        dsw.compute_output(0.0)
        self.assertEqual(dsr.compute_output(0.0)[0], 50.0)

    # =========================================================================
    # 10. SIGNAL ATTRIBUTES
    # =========================================================================

    def test_signal_attributes(self):
        ic = ICBlock("IC", initial_value=100.0)
        ic.inputs = [5.0]
        self.assertEqual(ic.compute_output(0.0)[0], 100.0)
        self.assertEqual(ic.compute_output(0.1)[0], 5.0)

        dtc = DataTypeConversionBlock("DTC", target_data_type="int32")
        dtc.inputs = [3.8]
        self.assertEqual(dtc.compute_output(0.0)[0], 3.0)

        probe = ProbeBlock("Probe")
        probe.inputs = [[1.0, 2.0, 3.0]]
        self.assertEqual(probe.compute_output(0.0)[0], 3.0)  # width 3

        uconv = UnitConversionBlock("UConv", input_unit="deg", output_unit="rad")
        uconv.inputs = [180.0]
        self.assertAlmostEqual(uconv.compute_output(0.0)[0], math.pi)

    # =========================================================================
    # 11. MODEL VERIFICATION
    # =========================================================================

    def test_model_verification(self):
        assertion = AssertionBlock("Assert")
        assertion.inputs = [1.0]
        self.assertEqual(assertion.compute_output(0.0)[0], 1.0)
        assertion.inputs = [0.0]
        self.assertEqual(assertion.compute_output(0.0)[0], 0.0)
        self.assertFalse(assertion.is_valid)

        chk_range = CheckStaticRangeBlock("ChkRange", min_val=-5.0, max_val=5.0)
        chk_range.inputs = [3.0]
        self.assertEqual(chk_range.compute_output(0.0)[0], 1.0)
        chk_range.inputs = [10.0]
        self.assertEqual(chk_range.compute_output(0.0)[0], 0.0)

        chk_res = CheckInputResolutionBlock("ChkRes", resolution=0.25)
        chk_res.inputs = [1.50]
        self.assertEqual(chk_res.compute_output(0.0)[0], 1.0)
        chk_res.inputs = [1.40]
        self.assertEqual(chk_res.compute_output(0.0)[0], 0.0)

    # =========================================================================
    # 12. STRINGS
    # =========================================================================

    def test_string_blocks(self):
        sc = StringConstantBlock("SC", string_value="Terminus")
        self.assertEqual(sc.compute_output(0.0)[0], "Terminus")

        sconcat = StringConcatenateBlock("SConcat", separator="_")
        sconcat.inputs = ["Simulink", "Engine"]
        self.assertEqual(sconcat.compute_output(0.0)[0], "Simulink_Engine")

        s_find = StringFindBlock("SFind", pattern="min")
        s_find.inputs = ["Terminus"]
        self.assertEqual(s_find.compute_output(0.0)[0], 4.0)

        s2d = StringToDoubleBlock("S2D")
        s2d.inputs = ["3.14159"]
        self.assertAlmostEqual(s2d.compute_output(0.0)[0], 3.14159)

        compose = ComposeStringBlock("Compose", format_string="Speed: {:.2f} RPM")
        compose.inputs = [1200.55]
        self.assertEqual(compose.compute_output(0.0)[0], "Speed: 1200.55 RPM")

    # =========================================================================
    # 13. USER FUNCTIONS & SUBSYSTEMS
    # =========================================================================

    def test_user_functions_and_subsystems(self):
        fcn = FcnBlock("Fcn", expression="u[0]**2 + sin(u[0])")
        fcn.inputs = [0.0]
        self.assertEqual(fcn.compute_output(0.0)[0], 0.0)

        mfunc = MATLABFunctionBlock("MFunc", func=lambda u, t: [u[0] * 3.0, u[0] + 1.0], num_outputs=2)
        mfunc.inputs = [4.0]
        out_mf = mfunc.compute_output(0.0)
        self.assertEqual(out_mf[0], 12.0)
        self.assertEqual(out_mf[1], 5.0)

        # For Iterator Subsystem
        for_sub = ForIteratorSubsystemBlock("ForSub", num_iterations=4, iter_fn=lambda i, acc: acc + 2)
        for_sub.inputs = [0.0]
        self.assertEqual(for_sub.compute_output(0.0)[0], 8.0)

        # Enabled Subsystem
        en_sub = EnabledSubsystemBlock("EnSub", forward_fn=lambda u, t: [u[1] * 10.0])
        en_sub.inputs = [0.0, 5.0]  # disabled
        self.assertEqual(en_sub.compute_output(0.0)[0], 0.0)
        en_sub.inputs = [1.0, 5.0]  # enabled
        self.assertEqual(en_sub.compute_output(0.0)[0], 50.0)

    # =========================================================================
    # 14. DASHBOARDS & INTERACTIVE CONTROLS
    # =========================================================================

    def test_dashboard_blocks(self):
        slider = SliderBlock("Slider", min_val=0.0, max_val=100.0, initial_value=25.0)
        self.assertEqual(slider.compute_output(0.0)[0], 25.0)
        slider.set_value(75.0)
        self.assertEqual(slider.compute_output(0.0)[0], 75.0)

        knob = KnobBlock("Knob", min_val=0.0, max_val=10.0, initial_value=3.0)
        self.assertEqual(knob.compute_output(0.0)[0], 3.0)

        toggle = ToggleSwitchBlock("Toggle", off_val=0.0, on_val=1.0, initial_state=False)
        self.assertEqual(toggle.compute_output(0.0)[0], 0.0)
        toggle.toggle()
        self.assertEqual(toggle.compute_output(0.0)[0], 1.0)

        gauge = GaugeBlock("Gauge", min_val=0.0, max_val=200.0)
        gauge.inputs = [150.0]
        self.assertEqual(gauge.compute_output(0.0)[0], 150.0)

        lamp = LampBlock("Lamp", thresholds=[0.0, 50.0, 100.0], colors=["green", "yellow", "red"])
        lamp.inputs = [75.0]
        self.assertEqual(lamp.compute_output(0.0)[0], "yellow")

    # =========================================================================
    # 15. MESSAGES AND QUEUES
    # =========================================================================

    def test_messages_and_queues(self):
        q = QueueBlock("Queue", capacity=10)
        # Push 42.0
        q.inputs = [42.0, 1.0, 0.0]
        q.compute_output(0.0)
        # Pop
        q.inputs = [0.0, 0.0, 1.0]
        out_q = q.compute_output(0.1)
        self.assertEqual(out_q[0], 42.0)
        self.assertEqual(out_q[1], 0.0)  # Length 0


if __name__ == "__main__":
    unittest.main()
