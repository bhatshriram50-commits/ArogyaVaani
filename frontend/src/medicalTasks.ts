export type MedicalTaskId = 'PNEUMONIA' | 'RETINOPATHY' | 'BRAIN_TUMOR';
export type MedicalTask = { taskId: MedicalTaskId; displayName: string; organ: string; modality: string; architecture: string; inputSize: [number, number]; classLabels: string[]; preprocessing: 'imagenet-rgb'; enabled: boolean };
export const fallbackTasks: MedicalTask[] = [
  { taskId: 'PNEUMONIA', displayName: 'Pneumonia', organ: 'LUNG', modality: 'CHEST_XRAY', architecture: 'densenet121', inputSize: [224, 224], classLabels: ['NORMAL', 'PNEUMONIA'], preprocessing: 'imagenet-rgb', enabled: true },
  { taskId: 'RETINOPATHY', displayName: 'Retinopathy', organ: 'EYE', modality: 'FUNDUS_IMAGE', architecture: 'densenet121', inputSize: [224, 224], classLabels: ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'], preprocessing: 'imagenet-rgb', enabled: true },
  { taskId: 'BRAIN_TUMOR', displayName: 'Brain Tumor', organ: 'BRAIN', modality: 'MRI', architecture: 'densenet121', inputSize: [224, 224], classLabels: ['glioma', 'meningioma', 'notumor', 'pituitary'], preprocessing: 'imagenet-rgb', enabled: true },
];
export const examinationLabel = (task: MedicalTask) => task.modality === 'CHEST_XRAY' ? 'Chest X-ray' : task.modality === 'FUNDUS_IMAGE' ? 'Retinal Image' : 'Brain MRI';
