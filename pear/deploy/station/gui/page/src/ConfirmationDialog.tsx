import {Home, Power, X} from 'lucide-react';
import type {HomePlan, ServerState} from './protocol';

export type DialogKind = 'home' | 'disable';

type Command = (name: string, data?: Record<string, unknown>) => Promise<boolean>;

type DialogProps = {
  kind: DialogKind | null;
  state: ServerState | null;
  command: Command;
  homeOpen: boolean;
  homeSide: string;
  onHomeSide: (side: string) => void;
  onClose: () => void;
};

/** Confirmation of disabling the motors, or of a Home move that the service has planned or is about to plan. */
export function ConfirmationDialog({kind, state, command, homeOpen, homeSide, onHomeSide, onClose}: DialogProps) {
  const plan = state?.home_plan;
  if (!kind && !plan) return null;

  function close() {
    onClose();
    if (plan) void command('home_confirm', {plan_id: plan.plan_id, confirm: false});
  }

  function title(): string {
    if (kind === 'disable') return 'Release motor power?';
    if (plan) return plan.open_grippers === false ? 'Confirm direct Home (keep grippers)' : 'Confirm direct Home';
    return homeOpen ? 'Return to home' : 'Return arms to home';
  }

  function body() {
    if (kind === 'disable') {
      return (
        <>
          <p>Support the arms and any held objects before releasing motor power. This ends the current task.</p>
          <button
            className="button danger wide"
            onClick={() => {
              onClose();
              void command('disable', {confirm: true});
            }}
          >
            Disable Motors
          </button>
        </>
      );
    }
    if (plan) return <HomePlanBody plan={plan} command={command} />;
    return (
      <HomeCheckBody
        state={state}
        command={command}
        homeOpen={homeOpen}
        homeSide={homeSide}
        onHomeSide={onHomeSide}
        onClose={onClose}
      />
    );
  }

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <button className="modal-close" aria-label="Close dialog" onClick={close}>
          <X size={20} />
        </button>
        <div className="modal-icon">{kind === 'disable' ? <Power size={24} /> : <Home size={24} />}</div>
        <h2>{title()}</h2>
        {body()}
      </div>
    </div>
  );
}

function HomePlanBody({plan, command}: {plan: HomePlan; command: Command}) {
  const sides = plan.active_sides ? `Selected arms: ${plan.active_sides.join(', ')} · ` : '';
  const seconds = plan.operation_seconds_estimate?.toFixed(1) ?? '—';
  return (
    <>
      {plan.open_grippers === false ? (
        <>
          <p>
            {sides}Grippers unchanged, arms return directly · about {seconds} seconds including settling.
          </p>
          <p>
            Clear the workspace. Collision checks are disabled. Grippers are neither opened nor closed, so held objects
            travel with the arms.
          </p>
        </>
      ) : (
        <>
          <p>
            {sides}Open grippers, then return directly · about {seconds} seconds including opening and settling.
          </p>
          <p>
            Clear the workspace and put down carried objects. Collision checks are disabled. Selected grippers open
            first and release held objects before the arms move.
          </p>
        </>
      )}
      <button
        className="button primary wide"
        onClick={() => command('home_confirm', {plan_id: plan.plan_id, confirm: true})}
      >
        Workspace clear — Begin Home
      </button>
    </>
  );
}

type HomeCheckBodyProps = Pick<DialogProps, 'state' | 'command' | 'homeOpen' | 'homeSide' | 'onHomeSide' | 'onClose'>;

function HomeCheckBody({state, command, homeOpen, homeSide, onHomeSide, onClose}: HomeCheckBodyProps) {
  const homeSides = state?.profile.home_sides ?? [];
  const keepGrippers = !!state?.profile.home_keep_grippers;
  return (
    <>
      <p>
        The current task will end. The home trajectory and joint limits are checked first.{' '}
        {homeOpen
          ? 'After confirmation, selected grippers open before the arms return home.'
          : 'After confirmation, only the arm joints return home; grippers keep their current opening.'}
      </p>
      {homeSides.length > 0 && (
        <label className="field-label">
          Arms
          <select aria-label="Home arms" value={homeSide} onChange={(e) => onHomeSide(e.target.value)}>
            {homeSides.map((side) => (
              <option key={side} value={side}>
                {side[0].toUpperCase() + side.slice(1)}
              </option>
            ))}
          </select>
        </label>
      )}
      <button
        className="button primary wide"
        onClick={() => {
          onClose();
          void command('home', {
            ...(homeSides.length ? {side: homeSide} : {}),
            ...(keepGrippers ? {open_grippers: homeOpen} : {}),
          });
        }}
      >
        Check Home
      </button>
    </>
  );
}
