import React from 'react';
import {Gauge} from 'lucide-react';
import type {Observation, RobotStatus} from './protocol';
import './ArmState.css';

type Values = Record<string, unknown>;
const LENGTHS = new Set(['x', 'y', 'z', 'width']);
const value = (name: string, v: unknown) =>
  typeof v === 'number' && Number.isFinite(v)
    ? `${v.toFixed(LENGTHS.has(name) ? 1 : 2)} ${LENGTHS.has(name) ? 'cm' : 'rad'}`
    : '—';

// Beside a single camera view: live joints of the robot's one arm (named by the robot profile,
// the gripper last), and what the model last saw in its own format.
export function ArmState({
  joints,
  robot,
  observed,
  format,
}: {
  joints: string[];
  robot?: RobotStatus;
  observed?: Observation['state'];
  format?: string;
}) {
  const position = Object.values(robot?.arms ?? {})[0]?.position;
  const seen = observed ? Object.values(observed).reduce((all: Values, part) => ({...all, ...part}), {}) : undefined;
  return (
    <div className="camera arm-state" aria-label="Arm state">
      <div className="camera-head">
        <span>
          <Gauge size={15} />
          Arm state
        </span>
        <span className="format-badge" title="Action format of the selected model">
          {format || '—'}
        </span>
      </div>
      <div className="arm-state-body">
        <section>
          <h3>Live joints</h3>
          {position ? (
            <dl>
              {joints.map((name, i) => (
                <React.Fragment key={name}>
                  <dt>{name}</dt>
                  <dd>{((position[i] * 180) / Math.PI).toFixed(1)}°</dd>
                </React.Fragment>
              ))}
              <dt>Gripper</dt>
              <dd>{Math.round(position[joints.length] * 100)}% open</dd>
            </dl>
          ) : (
            <p>Motors disabled</p>
          )}
        </section>
        <section>
          <h3 title="What the model saw, in its action format">Last observation</h3>
          {seen ? (
            <dl>
              {Object.entries(seen).map(([name, v]) => (
                <React.Fragment key={name}>
                  <dt>{name}</dt>
                  <dd>{value(name, v)}</dd>
                </React.Fragment>
              ))}
            </dl>
          ) : (
            <p>No observation in this task yet</p>
          )}
        </section>
      </div>
    </div>
  );
}
