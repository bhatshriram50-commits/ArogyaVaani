export type MedicalTaskId = 'PNEUMONIA' | 'RETINOPATHY' | 'BRAIN_TUMOR';

export type MedicalTask = {
  taskId: MedicalTaskId;
  displayName: string;
  organ: 'LUNG' | 'EYE' | 'BRAIN';
  modality: 'CHEST_XRAY' | 'FUNDUS_IMAGE' | 'MRI';
  architecture: string;
  inputSize: [number, number];
  /** Empty means the hospital-local dataset determines labels for its first model. */
  classLabels: string[];
  preprocessing: 'imagenet-rgb';
  enabled: boolean;
};

export const medicalTasks: Record<MedicalTaskId, MedicalTask> = {
  PNEUMONIA: { taskId: 'PNEUMONIA', displayName: 'Pneumonia', organ: 'LUNG', modality: 'CHEST_XRAY', architecture: 'densenet121', inputSize: [224, 224], classLabels: ['NORMAL', 'PNEUMONIA'], preprocessing: 'imagenet-rgb', enabled: true },
  RETINOPATHY: { taskId: 'RETINOPATHY', displayName: 'Retinopathy', organ: 'EYE', modality: 'FUNDUS_IMAGE', architecture: 'densenet121', inputSize: [224, 224], classLabels: ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'], preprocessing: 'imagenet-rgb', enabled: true },
  BRAIN_TUMOR: { taskId: 'BRAIN_TUMOR', displayName: 'Brain Tumor', organ: 'BRAIN', modality: 'MRI', architecture: 'densenet121', inputSize: [224, 224], classLabels: ['glioma', 'meningioma', 'notumor', 'pituitary'], preprocessing: 'imagenet-rgb', enabled: true },
};

export const taskFor = (value: unknown): MedicalTask | undefined => typeof value === 'string' ? medicalTasks[value as MedicalTaskId] : undefined;
export const enabledTaskFor = (value: unknown): MedicalTask | undefined => {
  const task = taskFor(value);
  return task?.enabled ? task : undefined;
};

export const examinationTypes = Object.values(medicalTasks).map(task => ({ modality: task.modality, label: task.modality === 'CHEST_XRAY' ? 'Chest X-ray' : task.modality === 'FUNDUS_IMAGE' ? 'Retinal Image' : 'Brain MRI', task: task.taskId }));
