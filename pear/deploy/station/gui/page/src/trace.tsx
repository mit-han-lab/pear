import {createRoot} from 'react-dom/client';
import {TracePage, type TraceData} from './TracePage';

const trace: TraceData = JSON.parse(document.getElementById('trace-data')!.textContent!);
document.title = `${trace.robot} trace · ${trace.task.instruction}`;
createRoot(document.getElementById('root')!).render(<TracePage trace={trace} />);
