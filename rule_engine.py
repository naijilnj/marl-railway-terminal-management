class StationRuleEngine:

    # POINT 1 
    def rule_point_1_line_clear(self, line_clear, bell_rung,
                                prescribed_beats, passenger_train,
                                station_announced, unauthorized_access):
        violations = []
        reward = 0

        if line_clear and not (bell_rung and prescribed_beats):
            violations.append("Line Clear without bell or prescribed beats")
            reward -= 10
        else:
            reward += 2

        if passenger_train and not station_announced:
            violations.append("Passenger train without station announcement")
            reward -= 5
        else:
            reward += 1

        if unauthorized_access:
            violations.append("Unauthorized access to equipment")
            reward -= 20
        else:
            reward += 3

        return len(violations) == 0, violations, reward

    # POINT 2 
    def rule_point_2_berthing(self, stopping_passenger_train,
                              received_on_platform,
                              fouling_cleared,
                              beyond_platform,
                              special_permission):
        violations = []
        reward = 0

        if stopping_passenger_train and not received_on_platform:
            violations.append("Stopping passenger train not on platform line")
            reward -= 15
        else:
            reward += 3

        if not fouling_cleared:
            violations.append("Movement authorized without fouling clearance")
            reward -= 15
        else:
            reward += 4

        if beyond_platform and not special_permission:
            violations.append("Beyond-platform berthing without permission")
            reward -= 15
        else:
            reward += 2

        return len(violations) == 0, violations, reward

    # POINT 3 
    def rule_point_3_crossing(self, first_arriving_on_platform,
                              second_train_speed,
                              backing,
                              hand_signal_used):
        violations = []
        reward = 0

        if not first_arriving_on_platform:
            violations.append("First arriving train not on platform line")
            reward -= 20
        else:
            reward += 2

        if second_train_speed > 15:
            violations.append("Second train exceeded speed limit during crossing")
            reward -= 20
        else:
            reward += 2

        if backing and not hand_signal_used:
            violations.append("Backing without hand signals")
            reward -= 20
        else:
            reward += 3

        return len(violations) == 0, violations, reward
    
    # -------------------------------------------------
    # S.R. 5.01(v) – Crossing of Two Passenger Trains
    # (Single Line Station with One Platform)
    # -------------------------------------------------
    def rule_sr_5_01_v_crossing_procedure(
        self,
        platform_height_mm,
        first_train_stops,
        second_train_stops,
        first_train_arrived,
        platform_length_sufficient,
        second_train_speed,
        hand_signal_used,
        loop_line_available
    ):
        """
        Returns:
            allowed (bool)
            procedure (str)
            violations (list)
        """

        violations = []
        procedure = None

        # Global safety constraints
        if second_train_speed > 15:
            violations.append("Speed exceeded 15 kmph during shunting/backing")

        if not hand_signal_used:
            violations.append("Hand signals not used during backing/shunting")

        # ---------------- CASE (a): BOTH TRAINS STOP ----------------
        if first_train_stops and second_train_stops:

            # Case (a)(1): Platform ≤ 455 mm
            if platform_height_mm <= 455:
                procedure = (
                    "First train on platform line, "
                    "second train on non-platform line; "
                    "both drawn ahead without infringing fouling marks"
                )

            # Case (a)(2): Platform ≥ 760 mm
            elif platform_height_mm >= 760:

                # Option 1: Shunt first train after boarding
                procedure = (
                    "First train admitted on platform line, "
                    "then shunted to non-platform line after passenger exchange; "
                    "second train admitted on platform line"
                )

                # Option 2: Pass second train and back after departure
                if not platform_length_sufficient:
                    procedure += (
                        " OR second train passed through and backed onto platform "
                        "after departure of first train under hand signals"
                    )

                # Option 3: Platform long enough for both
                if platform_length_sufficient:
                    procedure += (
                        " OR second train backed onto platform line while first train "
                        "is still standing, platform sufficient for both"
                    )

            else:
                violations.append(
                    "Invalid platform height for S.R. 5.01(v) conditions"
                )

        # ---------------- CASE (b): ONLY ONE TRAIN STOPS ----------------
        elif first_train_stops != second_train_stops:
            procedure = (
                "Stopping passenger train admitted first on platform line; "
                "non-stopping train passed through non-platform line"
            )

        # ---------------- CASE (c): NEITHER TRAIN STOPS ----------------
        else:
            procedure = (
                "First arriving train admitted on platform line; "
                "second train passed through non-platform line"
            )

        # ---------------- CASE (d): LOOP LINE PREFERENCE ----------------
        if loop_line_available and not second_train_stops:
            procedure += (
                " | First arriving train may be received on loop line "
                "with trailing points set to snag dead end/sand hump"
            )

        allowed = len(violations) == 0
        return allowed, procedure, violations


    # POINT 4 
    def rule_point_4_defects(self, defect_detected,
                             defect_reported,
                             emergency,
                             traffic_protected):
        violations = []
        reward = 0

        if defect_detected and not defect_reported:
            violations.append("Defect detected but not reported")
            reward -= 25
        else:
            reward += 5

        if emergency and not traffic_protected:
            violations.append("Emergency without traffic protection")
            reward -= 25
        else:
            reward += 5

        return len(violations) == 0, violations, reward

    # POINT 5 
    def rule_point_5_obstructed_reception(self, obstructed_line,
                                          signal_on,
                                          written_authority,
                                          calling_on_signal,
                                          stopped_before_obstruction,
                                          hand_signal_used):
        violations = []
        reward = 0

        if obstructed_line and not signal_on:
            violations.append("Signal taken OFF on obstructed line")
            reward -= 30

        if obstructed_line and not (written_authority or calling_on_signal):
            violations.append("No authority issued on obstructed line")
            reward -= 30
        else:
            reward += 3

        if not stopped_before_obstruction:
            violations.append("Train did not stop before obstruction")
            reward -= 30
        else:
            reward += 4

        if not hand_signal_used:
            violations.append("No hand signal used on obstructed reception")
            reward -= 20
        else:
            reward += 3

        return len(violations) == 0, violations, reward

    # POINT 6 
    def rule_point_6_departure(self, non_signalled_line,
                               written_authority,
                               points_set,
                               points_locked):
        violations = []
        reward = 0

        if non_signalled_line and not written_authority:
            violations.append("Departure from non-signalled line without authority")
            reward -= 40

        if not (points_set and points_locked):
            violations.append("Points not set and locked before departure")
            reward -= 40
        else:
            reward += 7

        return len(violations) == 0, violations, reward

    # POINT 7
    def rule_point_7_shunting(self, shunting,
                              controlled_by_signal,
                              speed,
                              passenger_vehicle,
                              sm_instruction):
        violations = []
        reward = 0

        if shunting and not controlled_by_signal:
            violations.append("Shunting not under signal/hand control")
            reward -= 20
        else:
            reward += 2

        if speed > 15:
            violations.append("Shunting speed exceeded limit")
            reward -= 20
        else:
            reward += 2

        if passenger_vehicle and not sm_instruction:
            violations.append("Passenger vehicle shunted without SM instruction")
            reward -= 20
        else:
            reward += 4

        return len(violations) == 0, violations, reward

    # POINT 8
    def rule_point_8_ctc(self, ctc_station,
                         shunting,
                         ctc_permission,
                         backing,
                         authorized):
        violations = []
        reward = 0

        if ctc_station and shunting and not ctc_permission:
            violations.append("Shunting without CTC permission")
            reward -= 25
        else:
            reward += 3

        if backing and not authorized:
            violations.append("Backing without authorization")
            reward -= 25
        else:
            reward += 3

        return len(violations) == 0, violations, reward

    # POINT 9
    def rule_point_9_running_line_obstruction(self, obstructed,
                                                sm_sanction,
                                                points_padlocked,
                                                register_entry):
        violations = []
        reward = 0

        if obstructed and not sm_sanction:
            violations.append("Running line obstructed without SM sanction")
            reward -= 30

        if obstructed and not (points_padlocked and register_entry):
            violations.append("Obstruction without padlocking or documentation")
            reward -= 30
        else:
            reward += 6

        return len(violations) == 0, violations, reward

    # POINT 10
    def rule_point_10_gradients(self, hand_shunting,
                                steep_gradient,
                                loose_shunting,
                                passenger_load,
                                dangerous_goods):
        violations = []
        reward = 0

        if hand_shunting and steep_gradient:
            violations.append("Hand shunting on steep gradient (PROHIBITED)")
            reward -= 50

        if loose_shunting and (passenger_load or dangerous_goods):
            violations.append("Loose shunting with passengers/dangerous goods")
            reward -= 50

        if not violations:
            reward += 3

        return len(violations) == 0, violations, reward
